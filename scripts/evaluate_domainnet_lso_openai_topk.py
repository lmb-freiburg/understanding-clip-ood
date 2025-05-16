import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from open_clip import get_tokenizer
from sklearn.metrics import f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset

from xclip.datasets import DomainNetCaptions, ImageNet, openai_imagenet_classes
from xclip.open_clip import OpenCLIP
from xclip.zero_shot import OpenAIZeroShotClassifier


def get_data(
    clip: OpenCLIP, dataset: Dataset, keys: list[str], num_workers: int, batch_size: int = 250
) -> dict[str, torch.Tensor]:
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip.to(device)
    clip.eval()

    data = {'img_feat': [], **{key: [] for key in keys}}
    for batch in dataloader:
        with torch.inference_mode():
            # compute image and text features
            data['img_feat'].append(F.normalize(clip.encode_image(batch[0].half().to(device))))

        for i, key in enumerate(keys):
            data[key].append(batch[i + 1])

    data = {k: torch.cat(v) for k, v in data.items()}
    return data


def evaluate_model(
    clip: OpenCLIP,
    tokenizer,
    data: dict,
    class_to_idx: dict[str, int],
    domain: str,
    domainnet_classes: dict[int, str],
    domain_invariant: bool,
    k: int,
    **data_kwargs,
) -> tuple[dict[str, dict[str, dict[str, float]]], tuple]:
    res = {
        'imagenet-val': {f'top-{k}-accuracy': {}, 'f1-score': {}},
        'domainnet-val': {
            f'top-{k}-accuracy': {},
            'f1-score': {},
            'num-samples': {},
        },
    }

    # retrieve train/test image features and labels for val
    val_data = get_data(clip, data['val'], ['clss'], batch_size=250, **data_kwargs)
    img_feat = val_data['img_feat']

    # create zero-shot classifier and predict
    zs_classifier = OpenAIZeroShotClassifier(clip, tokenizer, openai_imagenet_classes, domain_invariant)
    zs_logits = zs_classifier.predict_from_features(img_feat, return_scores=True)['pred'].cpu()
    zs_pred = zs_classifier.predict_from_features(img_feat)['pred'].cpu()

    # compute zero-shot accuracies
    res['imagenet-val'][f'top-{k}-accuracy']['total'] = top_k_accuracy_score(
        val_data['clss'].numpy(), zs_logits, k=k, labels=np.arange(1000)
    )
    res['imagenet-val']['f1-score']['total'] = f1_score(val_data['clss'].numpy(), zs_pred, average='macro')
    val_labels = val_data['clss'].numpy()
    val_pred = zs_pred.numpy()

    # retrieve train/test image features and labels
    domain_data = get_data(clip, data['domain'], ['clss'], batch_size=250, **data_kwargs)
    img_feat = domain_data['img_feat']

    domain_to_idx = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}
    domain_ids = np.array([domain_to_idx[sample[0].split('/')[-3]] for sample in data['domain'].samples])
    assert np.unique(domain_ids).size == 2  # real and the domain

    # create zero-shot classifier and predict
    zs_classifier = OpenAIZeroShotClassifier(clip, tokenizer, domainnet_classes, domain_invariant)
    zs_logits = zs_classifier.predict_from_features(img_feat, return_scores=True)['pred'].cpu()
    zs_pred = zs_classifier.predict_from_features(img_feat)['pred'].cpu()

    domain_labels = domain_data['clss'].numpy()
    domain_logits = zs_logits.numpy()
    domain_pred = zs_pred.numpy()

    for dom in [domain, 'real']:
        dom_mask = domain_ids == domain_to_idx[dom]
        kw_labels = domain_labels[dom_mask]
        assert kw_labels.size > 0

        # also compute zero-shot accuracies for the ood class
        gen_logits = domain_logits[dom_mask]
        gen_pred = domain_pred[dom_mask]
        lso_mask = np.isin(kw_labels, list(class_to_idx.values()))
        assert not np.all(lso_mask)

        res['domainnet-val'][f'top-{k}-accuracy'][f'{dom}-lso-ood'] = top_k_accuracy_score(
            kw_labels[lso_mask], gen_logits[lso_mask], k=k, labels=np.arange(345)
        )
        res['domainnet-val'][f'top-{k}-accuracy'][f'{dom}-lso-id'] = top_k_accuracy_score(
            kw_labels[~lso_mask], gen_logits[~lso_mask], k=k, labels=np.arange(345)
        )
        res['domainnet-val']['f1-score'][f'{dom}-lso-ood'] = f1_score(
            kw_labels[lso_mask],
            gen_pred[lso_mask],
            average='macro',
            labels=np.unique(kw_labels[lso_mask]),
        )
        res['domainnet-val']['f1-score'][f'{dom}-lso-id'] = f1_score(
            kw_labels[~lso_mask],
            gen_pred[~lso_mask],
            average='macro',
            labels=np.unique(kw_labels[~lso_mask]),
        )
        res['domainnet-val']['num-samples'][f'{dom}-lso-ood'] = len(kw_labels[lso_mask])
        res['domainnet-val']['num-samples'][f'{dom}-lso-id'] = len(kw_labels[~lso_mask])

        # also compute accuracies for each individual class
        for cls in class_to_idx:
            cls_mask = kw_labels == class_to_idx[cls]
            assert not np.all(cls_mask)

            res['domainnet-val'][f'top-{k}-accuracy'][f'{dom}-{cls}-ood'] = top_k_accuracy_score(
                kw_labels[cls_mask], gen_logits[cls_mask], k=k, labels=np.arange(345)
            )
            res['domainnet-val']['num-samples'][f'{dom}-{cls}-ood'] = len(kw_labels[cls_mask])

        id_accs = []
        ood_accs = []
        for label, cls in domainnet_classes.items():
            if cls in class_to_idx:
                assert label == class_to_idx[cls]
            cls_mask = kw_labels == label
            assert not np.all(cls_mask)

            if not np.any(cls_mask):
                # one class for painting is missing
                assert domain == 'painting'
                continue

            if cls in class_to_idx:
                ood_accs.append(
                    top_k_accuracy_score(kw_labels[cls_mask], gen_logits[cls_mask], k=k, labels=np.arange(345))
                )
            else:
                id_accs.append(
                    top_k_accuracy_score(kw_labels[cls_mask], gen_logits[cls_mask], k=k, labels=np.arange(345))
                )

        num_id_classes = 330 if dom != 'painting' else 329
        assert len(id_accs) == num_id_classes, f'{dom=} {len(id_accs)=}'
        assert len(ood_accs) == 15, f'{dom=} {len(ood_accs)=}'

        res['domainnet-val'][f'top-{k}-accuracy'][f'{dom}-lso-unweighted-id'] = np.mean(id_accs)
        res['domainnet-val'][f'top-{k}-accuracy'][f'{dom}-lso-unweighted-ood'] = np.mean(ood_accs)

    return res, (val_labels, val_pred, domain_labels, domain_pred, domain_ids)


def serialize_predictions(predictions: tuple, out_path: str) -> None:
    val_labels, val_pred, domain_labels, domain_pred, domain_ids = list(zip(*predictions))

    assert np.all([vl == val_labels[0] for vl in val_labels])
    assert np.all([sl == domain_labels[0] for sl in domain_labels])
    assert np.all([di == domain_ids[0] for di in domain_ids])

    val_labels = np.array(val_labels[0])
    val_pred = np.array(val_pred)
    domain_labels = np.array(domain_labels[0])
    domain_pred = np.array(domain_pred)
    domain_ids = np.array(domain_ids[0])

    np.save(os.path.join(out_path, 'val_labels.npy'), val_labels)
    np.save(os.path.join(out_path, 'val_pred.npy'), val_pred)
    np.save(os.path.join(out_path, 'domain_labels.npy'), domain_labels)
    np.save(os.path.join(out_path, 'domain_pred.npy'), domain_pred)
    np.save(os.path.join(out_path, 'domain_ids.npy'), domain_ids)


def main(args: argparse.Namespace) -> None:
    # extract step count from checkpoint filename
    def epoch_or_step_from_ckpt_file(filename: str) -> int:
        filename = os.path.basename(filename)
        begin = filename.find('step') + 5 if 'step' in filename else filename.find('epoch') + 6
        end = filename.find('.')
        return int(filename[begin:end])

    # retrieve and sort checkpoint files
    ckpt_files = args.ckpt_files
    ckpt_files = sorted(ckpt_files, key=epoch_or_step_from_ckpt_file)
    steps = [epoch_or_step_from_ckpt_file(file) for file in ckpt_files]
    results_per_step = []

    # load datasets and tokenizers
    *_, preprocess_val = OpenCLIP.from_pretrained(args.model)
    tokenizer = get_tokenizer(args.model)
    exclude_domains = [
        d for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch'] if args.domain and d != args.domain
    ]
    data = {
        'val': ImageNet(args.imagenet_path, split='val', transform=preprocess_val),
        'domain': DomainNetCaptions(
            args.domainnet_path, 'val', transform=preprocess_val, exclude_domains=exclude_domains
        ),
    }

    domainnet_classes = {}
    for path, label, _ in data['domain'].samples:
        *_, clss, _ = path.split('/')
        domainnet_classes[label] = clss.replace('_', ' ')

    class_to_idx = {
        'aircraft carrier': 0,
        'axe': 11,
        'banana': 13,
        'barn': 15,
        'bed': 25,
        'candle': 58,
        'lion': 174,
        'mountain': 190,
        'necklace': 197,
        'penguin': 218,
        'pizza': 225,
        'saxophone': 250,
        'television': 305,
        'tractor': 319,
        'traffic light': 320,
    }
    for cls, label in class_to_idx.items():
        assert domainnet_classes[label] == cls, f'{domainnet_classes[label]=} {cls=}'

    if not args.domain:
        data['domain'].samples = [
            sample for sample in data['domain'].samples if domainnet_classes[sample[1]] == args.cls
        ]

    print(f'Excluded {exclude_domains=}')

    for step, ckpt_file in tqdm.tqdm(list(zip(steps, ckpt_files)), desc='Evaluating models'):
        assert step == epoch_or_step_from_ckpt_file(ckpt_file)
        clip, *_ = OpenCLIP.from_pretrained(args.model, ckpt_path=ckpt_file)
        results_per_step.append(
            evaluate_model(
                clip,
                tokenizer,
                data,
                class_to_idx,
                args.domain,
                domainnet_classes,
                args.domain_invariant,
                num_workers=args.num_workers,
                k=args.top_k,
            )
        )

    results_per_step, predictions = list(zip(*results_per_step))
    serialize_predictions(predictions, args.out_path)

    k = args.top_k
    results = {
        'steps': steps,
        'classes': list(class_to_idx.keys()),
        'domain': args.domain,
        'imagenet-val': {f'top-{k}-accuracy': {}, 'f1-score': {}},
        'domainnet-val': {
            f'top-{k}-accuracy': {},
            'f1-score': {},
            'num-samples': {},
        },
    }
    for result in results_per_step:
        for metric in result:
            for split in result[metric]:
                for feature in result[metric][split]:
                    val = result[metric][split][feature]
                    try:
                        results[metric][split][feature].append(val)
                    except KeyError:
                        results[metric][split][feature] = [val]

    with open(os.path.join(args.out_path, 'results_topk.json'), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure CLIP models to evaluate.')
    parser.add_argument('--model', type=str, required=True, help='CLIP model type')
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['clipart', 'infograph', 'painting', 'quickdraw', 'sketch'],
        help='domain to evaluate',
    )
    parser.add_argument('--ckpt_files', type=str, nargs='+', help='checkpoints to evaluate')
    parser.add_argument('--out_path', type=str, required=True, help='output directory for results.json')
    parser.add_argument('--imagenet_path', type=str, required=True, help='path to imagenet directory')
    parser.add_argument('--domainnet_path', type=str, required=True, help='path to domainnet directory')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for dataloader')
    parser.add_argument('--domain_invariant', action='store_true', help='use domain invariant templates')
    parser.add_argument('--top_k', type=int, default=5, help='top k classes to evaluate')

    args = parser.parse_args()
    main(args)
