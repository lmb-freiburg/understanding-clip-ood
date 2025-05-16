import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sparse_autoencoder import SparseAutoencoder
from torch.utils.data import DataLoader, Dataset

from xclip.datasets import DomainNetCaptions
from xclip.open_clip import OpenCLIP


class SAE:
    def __init__(self, sae_path, input_dim=1024, expansion_factor=4, n_components=1, device='cpu'):
        n_learned_features = int(input_dim * expansion_factor)
        self.autoencoder = SparseAutoencoder(
            n_input_features=input_dim,
            n_learned_features=n_learned_features,
            n_components=n_components,
        ).to(device)

        step = 'final'
        # step = 100388864
        state_dict_path = os.path.join(sae_path, 'checkpoints', f'sparse_autoencoder_{step}.pt')
        self.autoencoder.load_state_dict(torch.load(state_dict_path, map_location=device))

        with open(os.path.join(sae_path, 'concepts', 'concept_names.csv'), 'r') as f:
            self.concept_names = [line.split(',')[1].strip() for line in f.readlines()]

    def get_concepts_from_features(self, x):
        concepts, _ = self.autoencoder.forward(x)
        concepts = concepts.squeeze(1)
        return concepts


def get_data(
    clip: OpenCLIP, dataset: Dataset, keys: list[str], num_workers: int, batch_size: int = 250
) -> dict[str, torch.Tensor]:
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip.to(device)
    clip.eval()

    data = {'img_feat': [], **{key: [] for key in keys}}
    for batch in tqdm.tqdm(dataloader, desc='Computing image features'):
        with torch.inference_mode():
            # compute image and text features
            data['img_feat'].append(F.normalize(clip.encode_image(batch[0].half().to(device))))

        for i, key in enumerate(keys):
            data[key].append(batch[i + 1])

    data = {k: torch.cat(v) for k, v in data.items()}
    return data


def pct_shared(hist_a, hist_b, k=10):
    a = torch.topk(torch.tensor(hist_a), k=k)[1]
    b = torch.topk(torch.tensor(hist_b), k=k)[1]
    a = set(a.numpy())
    b = set(b.numpy())
    return len(a.intersection(b)) / k


def mpct_shared(hist_a, hist_b, ks=[5, 10, 15, 20]):
    return sum([pct_shared(hist_a, hist_b, k=k) for k in ks]) / len(ks)


def evaluate_feature_sharing(
    sae: SAE,
    img_feat: torch.Tensor,
    domain_labels: np.ndarray,
    domain_ids: np.ndarray,
    class_to_idx: dict[str, int],
    domain_to_idx: dict[str, int],
    domain: str,
    out_path: str,
) -> dict[str, list[np.ndarray]]:
    histograms_top20 = {}
    for cls in class_to_idx:
        cls_idx = class_to_idx[cls]
        cls_mask = domain_labels == cls_idx

        histograms_top20[cls] = [np.zeros(4096) for _ in np.unique(domain_ids)]
        for domain in np.unique(domain_ids):
            domain_mask = domain_ids == domain
            mask = cls_mask & domain_mask

            cls_dom_img_feat = img_feat[mask]
            concepts = sae.get_concepts_from_features(cls_dom_img_feat)

            for concept in concepts:
                values, indices = concept.squeeze().topk(20)
                histograms_top20[cls][domain][indices[values > 0]] += 1

    domain_to_idx = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}
    for cls in class_to_idx:
        for domain in domain_to_idx:
            np.save(os.path.join(out_path, f'{cls}_{domain}_hist.npy'), histograms_top20[cls][domain_to_idx[domain]])

    results = {}
    for cls in class_to_idx:
        results[cls] = {
            'mpct_shared@20': [
                [mpct_shared(hist_a, hist_b, ks=[5, 10, 15, 20]) for hist_b in histograms_top20[cls]]
                for hist_a in histograms_top20[cls]
            ],
        }

    results['avg'] = {
        'mpct_shared@20': [
            [
                [mpct_shared(hist_a, hist_b, ks=[5, 10, 15, 20]) for hist_b in histograms_top20[cls]]
                for hist_a in histograms_top20[cls]
            ]
            for cls in class_to_idx
        ],
    }

    results['score'] = {}
    for k, v in results['avg'].items():
        m = np.mean(np.array(v), axis=0)
        results['avg'][k] = m.tolist()

        assert np.array_equal(m[domain_to_idx[domain]], m[:, domain_to_idx[domain]])
        scores = m[domain_to_idx[domain]]
        assert scores[domain_to_idx[domain]] == 1
        scores = np.delete(scores, domain_to_idx[domain])
        results['score'][k] = np.mean(scores).item()

    with open(os.path.join(out_path, 'feature-sharing.json'), 'w') as f:
        json.dump(results, f)

    return histograms_top20


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)

    clip, _, preprocess_val = OpenCLIP.from_pretrained(
        'RN50', ckpt_path=os.path.join(args.model_path, 'checkpoints', 'epoch_32.pt')
    )
    sae = SAE(os.path.join(args.model_path, 'sae'))

    dataset = DomainNetCaptions(args.domainnet_path, 'val', transform=preprocess_val)

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

    domain_to_idx = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}

    domain_data = get_data(clip, dataset, ['label'], batch_size=256, num_workers=args.num_workers)
    img_feat = domain_data['img_feat'].cpu()
    domain_labels = domain_data['label'].numpy()

    domain_to_idx = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}
    domain_ids = np.array([domain_to_idx[sample[0].split('/')[-3]] for sample in dataset.samples])
    assert np.unique(domain_ids).size == 6

    out_path = os.path.join(args.model_path, 'sae', 'features')
    os.makedirs(out_path, exist_ok=True)
    evaluate_feature_sharing(
        sae, img_feat, domain_labels, domain_ids, class_to_idx, domain_to_idx, args.domain, out_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory.')
    parser.add_argument('--domainnet_path', type=str, required=True, help='Path to the DomainNet dataset.')
    parser.add_argument('--domain', type=str, required=True, help='Domain to evaluate.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for data loading.')
    main(parser.parse_args())
