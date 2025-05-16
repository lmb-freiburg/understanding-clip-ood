import argparse
import os
import random

import tqdm

from xclip.datasets import ImageNetSketch

sketch_templates = [
    '{}.',
    'a {}.',
    'the {}.',
    '{} drawing.',
    'drawing of a {}.',
    'drawing of the {}.',
    'a {} drawing.',
    'a drawing of a {}.',
    'a drawing of the {}.',
    '{} sketch.',
    'sketch of a {}.',
    'sketch of the {}.',
    'a {} sketch.',
    'a sketch of a {}.',
    'a sketch of the {}.',
    '{} image.',
    'image of a {}.',
    'image of the {}.',
    'a {} image.',
    'an image of a {}.',
    'an image of the {}.',
]


def get_caption(name: str) -> str:
    # choose random template
    template = random.choice(sketch_templates)
    assert template[-1] == '.'
    template = template if random.random() < 0.5 else template[:-1]  # randomly drop the full stop

    # insert class name into template and caption as json
    return template.format(name)


def main(args: argparse.Namespace):
    random.seed(args.seed)

    assert os.path.isdir(os.path.join(args.imagenet_path, 'sketch')), (
        f"Expected {args.imagenet_path} to contain a directory 'sketch'."
    )
    dataset = ImageNetSketch(args.imagenet_path, transform=None)

    with open(os.path.join(args.imagenet_path, 'in-sketch-captions.tsv'), 'w') as f:
        f.write('filepath\ttitle\n')
        for path, label in tqdm.tqdm(dataset.samples, desc='Generating captions'):
            path = os.path.join(args.imagenet_path, path)
            path = os.path.abspath(path)
            assert os.path.isfile(path), f'Expected file {path} to exist.'

            caption = get_caption(dataset.class_labels[label])
            f.write(f'{path}\t{caption}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure ImageNet-Sketch caption generation.')
    parser.add_argument('--imagenet_path', type=str, help='path to imagenet directory (containing sketch)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    main(args)
