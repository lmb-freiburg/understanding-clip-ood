import argparse
import json
import os

import numpy as np
import open_clip
import torch
import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

from xclip.datasets import DomainNetCaptions
from xclip.learner import ImageNetCaptionsLearner


def get_data(
    classifier: ImageNetCaptionsLearner, dataset: Dataset, num_workers: int, batch_size: int = 125
) -> dict[str, torch.Tensor]:
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier.to(device)
    classifier.eval()

    data = {'pred': [], 'labels': []}
    for batch in dataloader:
        img, label = batch
        with torch.inference_mode():
            # logits have shape (batch_size, 1345) where the last 345 classes are the domainnet classes
            logits = classifier(img.to(device))[:, 1000:]
            assert logits.shape[-1] == 345, logits.shape
            data['pred'].append(logits.argmax(dim=-1))
            data['labels'].append(label)

    data = {k: torch.cat(v) for k, v in data.items()}
    return data


def evaluate_model(
    classifier: ImageNetCaptionsLearner,
    data: dict,
    class_to_idx: dict[str, int],
    domain: str,
    domainnet_classes: dict[int, str],
    **data_kwargs,
) -> dict[str, dict[str, dict[str, float]]]:
    res = {
        'domainnet-val': {'accuracy': {}, 'num-samples': {}},
    }

    # retrieve train/test image features and labels
    domain_data = get_data(classifier, data['domain'], batch_size=250, **data_kwargs)
    domain_labels = domain_data['labels'].numpy()
    domain_pred = domain_data['pred'].cpu().numpy()

    domain_to_idx = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}
    domain_ids = np.array([domain_to_idx[sample[0].split('/')[-3]] for sample in data['domain'].samples])
    assert np.unique(domain_ids).size == 2  # real and the domain

    for dom in [domain, 'real']:
        dom_mask = domain_ids == domain_to_idx[dom]
        kw_labels = domain_labels[dom_mask]
        assert kw_labels.size > 0

        # also compute zero-shot accuracies for the ood class
        gen_pred = domain_pred[dom_mask]
        lso_mask = np.isin(kw_labels, list(class_to_idx.values()))
        assert not np.all(lso_mask)

        res['domainnet-val']['accuracy'][f'{dom}-lso-ood'] = accuracy_score(kw_labels[lso_mask], gen_pred[lso_mask])
        res['domainnet-val']['accuracy'][f'{dom}-lso-id'] = accuracy_score(kw_labels[~lso_mask], gen_pred[~lso_mask])
        res['domainnet-val']['num-samples'][f'{dom}-lso-ood'] = len(kw_labels[lso_mask])
        res['domainnet-val']['num-samples'][f'{dom}-lso-id'] = len(kw_labels[~lso_mask])

        # also compute accuracies for each individual class
        for cls in class_to_idx:
            cls_mask = kw_labels == class_to_idx[cls]
            assert not np.all(cls_mask)

            res['domainnet-val']['accuracy'][f'{dom}-{cls}-ood'] = accuracy_score(
                kw_labels[cls_mask], gen_pred[cls_mask]
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
                ood_accs.append(accuracy_score(kw_labels[cls_mask], gen_pred[cls_mask]))
            else:
                id_accs.append(accuracy_score(kw_labels[cls_mask], gen_pred[cls_mask]))

        num_id_classes = 330 if dom != 'painting' else 329
        assert len(id_accs) == num_id_classes, f'{dom=} {len(id_accs)=}'
        assert len(ood_accs) == 15, f'{dom=} {len(ood_accs)=}'

        res['domainnet-val']['accuracy'][f'{dom}-lso-unweighted-id'] = np.mean(id_accs)
        res['domainnet-val']['accuracy'][f'{dom}-lso-unweighted-ood'] = np.mean(ood_accs)

    return res


def serialize_predictions(predictions: tuple, out_path: str) -> None:
    val_labels, val_pred, domain_labels, domain_pred, domain_ids, domain_kw_labels, domain_kw_preds = list(
        zip(*predictions)
    )

    assert np.all([vl == val_labels[0] for vl in val_labels])
    assert np.all([sl == domain_labels[0] for sl in domain_labels])
    assert np.all([di == domain_ids[0] for di in domain_ids])
    for kw_labels in domain_kw_labels:
        for dom in kw_labels:
            assert np.all(kw_labels[dom] == domain_kw_labels[0][dom])

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

    for dom in domain_kw_labels[0]:
        kw_labels = np.array(domain_kw_labels[0][dom])
        kw_preds = np.array([domain_kw_preds[e][dom] for e in range(len(domain_kw_preds))])

        if kw_labels.size > 0:
            np.save(os.path.join(out_path, f'{dom}_kw_labels.npy'), kw_labels)
            np.save(os.path.join(out_path, f'{dom}_kw_preds.npy'), kw_preds)


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
    preprocess_val = open_clip.image_transform(image_size=224, is_train=False)
    exclude_domains = [
        d for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch'] if args.domain and d != args.domain
    ]
    data = {
        'domain': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,  # type: ignore
            exclude_domains=exclude_domains,
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

    print(f'Excluded {exclude_domains=}')

    for step, ckpt_file in tqdm.tqdm(list(zip(steps, ckpt_files)), desc='Evaluating models'):
        assert step == epoch_or_step_from_ckpt_file(ckpt_file)
        classifier = ImageNetCaptionsLearner(args.model, lr=0.0, num_classes=1345)
        state_dict = torch.load(os.path.join(ckpt_file), map_location='cpu')['state_dict']
        classifier.load_state_dict(state_dict)
        results_per_step.append(
            evaluate_model(classifier, data, class_to_idx, args.domain, domainnet_classes, num_workers=args.num_workers)
        )

    results = {
        'steps': steps,
        'classes': list(class_to_idx.keys()),
        'domain': args.domain,
        'domainnet-val': {'accuracy': {}, 'num-samples': {}},
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

    with open(os.path.join(args.out_path, 'results.json'), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure supervised models to evaluate.')
    parser.add_argument('--model', type=str, default='rn50-clip', help='supervised model type')
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['clipart', 'infograph', 'painting', 'quickdraw', 'sketch'],
        help='domain to evaluate',
    )
    parser.add_argument('--ckpt_files', type=str, nargs='+', help='checkpoints to evaluate')
    parser.add_argument('--out_path', type=str, required=True, help='output directory for results.json')
    parser.add_argument('--domainnet_path', type=str, required=True, help='path to domainnet directory')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for dataloader')

    args = parser.parse_args()
    main(args)
