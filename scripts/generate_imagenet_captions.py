import argparse
import json
import os
import random
import warnings

import tqdm
from textacy import preprocessing


def create_caption_from_sample(sample: dict) -> str:
    title = sample['title']
    tags = ' '.join(sample['tags'])
    desc = sample['description']

    caption = '; '.join([part for part in [title, tags, desc] if part != ''])
    caption = preprocessing.normalize.whitespace(caption)
    # caption = preprocessing.remove.html_tags(caption)
    # caption = preprocessing.replace.urls(caption)
    # caption = preprocessing.replace.emails(caption)
    return caption


def main(args: argparse.Namespace):
    random.seed(args.seed)

    with open(args.imagenet_captions_path) as f:
        captions = json.load(f)

    with open(args.imagenet_captions_split_path) as f:
        train_val_split = json.load(f)
    train_samples = set(train_val_split['train'])
    val_samples = set(train_val_split['val'])

    train_samples_tsv = ['filepath\ttitle\n']
    val_samples_tsv = ['filepath\ttitle\n']
    skipped = 0
    for sample in tqdm.tqdm(captions, desc='Generating captions'):
        path = os.path.abspath(os.path.join(args.imagenet_train_path, sample['wnid'], sample['filename']))
        assert os.path.isfile(path), f'Expected file {path} to exist.'

        caption = create_caption_from_sample(sample)
        caption = caption.replace('\n', ' ')  # remove newlines since in TSV we need only one line per sample
        with open(f'{os.path.splitext(path)[0]}.json', 'w') as f:
            json.dump({'caption': caption}, f)
        if sample['filename'] in train_samples:
            train_samples_tsv.append(f'{path}\t{caption}\n')
        elif sample['filename'] in val_samples:
            val_samples_tsv.append(f'{path}\t{caption}\n')
        else:
            warnings.warn(
                f'Filename {sample["filename"]} not found in train or val samples. File will be skipped. This likely '
                'means that the "imagenet_captions.json" file has changed since our work. If this happens frequently, '
                'consider adjusting this behavior or creating your own train-val split.'
            )
            skipped += 1

    print(f'Skipped {skipped} out of {len(captions)} samples.')

    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, 'in-captions-train.tsv'), 'w') as f:
        f.writelines(train_samples_tsv)

    with open(os.path.join(args.out_path, 'in-captions-val.tsv'), 'w') as f:
        f.writelines(val_samples_tsv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure ImageNet caption generation.')
    parser.add_argument('--imagenet_train_path', type=str, help='path to imagenet train directory')
    parser.add_argument(
        '--imagenet_captions_path',
        type=str,
        default='data/imagenet_captions.json',
        help='path to the imagenet captions json file',
    )
    parser.add_argument(
        '--imagenet_captions_split_path',
        type=str,
        default='data/imagenet_captions_train_val_split.json',
        help='path to the imagenet captions train val split json file',
    )
    parser.add_argument('--out_path', type=str, default='data/indices', help='path to store the tsv files')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    main(args)
