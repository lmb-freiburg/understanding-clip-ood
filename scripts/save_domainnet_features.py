import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset

from xclip.datasets import DomainNetCaptions
from xclip.open_clip import OpenCLIP


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


def evaluate_model(clip: OpenCLIP, data: dict, **data_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # retrieve train/test image features and labels
    domain_data = get_data(clip, data['domain'], ['label'], batch_size=256, **data_kwargs)
    img_feat = domain_data['img_feat'].cpu().numpy()

    domain_to_idx = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}
    domain_ids = np.array([domain_to_idx[sample[0].split('/')[-3]] for sample in data['domain'].samples])
    assert np.unique(domain_ids).size == 6

    domain_labels = domain_data['label'].numpy()

    return img_feat, domain_labels, domain_ids


def main(args: argparse.Namespace) -> None:
    # load datasets and tokenizers
    *_, preprocess_val = OpenCLIP.from_pretrained(args.model)
    data = {'domain': DomainNetCaptions(args.domainnet_path, 'val', transform=preprocess_val)}

    img_feats = []
    domain_labels = None
    domain_ids = None
    for ckpt_file in tqdm.tqdm(args.ckpt_files, desc='Aggregating model features'):
        clip, *_ = OpenCLIP.from_pretrained(args.model, ckpt_path=ckpt_file)
        img_feat, labels, ids = evaluate_model(clip, data, num_workers=args.num_workers)
        img_feats.append(img_feat)

        if domain_labels is None:
            domain_labels = labels
            domain_ids = ids
        else:
            assert np.all(domain_labels == labels)
            assert np.all(domain_ids == ids)

    img_feat = np.stack(img_feats)
    assert isinstance(domain_labels, np.ndarray)
    assert isinstance(domain_ids, np.ndarray)

    np.save(os.path.join(args.out_path, 'img_feat.npy'), img_feat)
    np.save(os.path.join(args.out_path, 'domain_labels.npy'), domain_labels)
    np.save(os.path.join(args.out_path, 'domain_ids.npy'), domain_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure CLIP models to evaluate.')
    parser.add_argument('--model', type=str, required=True, help='CLIP model type')
    parser.add_argument('--ckpt_files', type=str, nargs='+', required=True, help='checkpoint to evaluate')
    parser.add_argument('--out_path', type=str, required=True, help='output directory for results.json')
    parser.add_argument('--domainnet_path', type=str, required=True, help='path to domainnet directory')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for dataloader')

    args = parser.parse_args()
    main(args)
