import argparse
import json
import os
import random

import tqdm

from xclip.datasets import openai_imagenet_classes


def main(args: argparse.Namespace):
    # Add ImageNet-Sketch captions to the following data mixtures:
    #     - natural-only (resulting in a CG low-diversity equivalent with IN instead of DN sketches)
    #     - leave-out-domain (resulting in a CG high-diversity equivalent with IN instead of DN sketches)
    #     - CG high-diversity (resulting in a CG high-diversity equivalent with a mix of both IN and DN sketches)

    for name in tqdm.tqdm(
        [
            'combined-captions-train-lso-real-only',
            'combined-captions-train-lso-cipqr-nosketchclasses',
            'combined-captions-train-lso-cipqrs-nosketchclasses',
        ]
    ):
        tsv_path = os.path.join(args.indices_path, f'{name}.tsv')

        with open(tsv_path, 'r') as f:
            samples = f.readlines()

        assert samples[0] == 'filepath\ttitle\n'
        samples = samples[1:]

        captions = [sample for sample in samples if 'captions' in sample.split('\t')[0]]
        domainnet = [sample for sample in samples if 'domainnet' in sample.split('\t')[0]]
        assert set(samples) == set(captions) | set(domainnet)
        assert set(captions) & set(domainnet) == set()

        with open(os.path.join(args.imagenet_path, 'in-sketch-captions.tsv'), 'r') as f:
            sketches = f.readlines()

        assert sketches[0] == 'filepath\ttitle\n'
        sketches = sketches[1:]

        class_labels = {i: cls_name for i, cls_name in enumerate(openai_imagenet_classes)}
        class_names = openai_imagenet_classes

        with open(args.class_mapping_path, 'r') as f:
            in_to_dn = json.load(f)

        shared_classes = []
        for clss in in_to_dn.values():
            if clss is not None:
                shared_classes.extend(clss)
        shared_class_names = [class_labels[c] for c in shared_classes]

        non_shared_classes = list(set(class_names) - set(shared_class_names))
        assert len(non_shared_classes) == 550

        sketches_filtered = []
        for sketch in sketches:
            _, caption = sketch.split('\t')

            # skip any ImageNet classes that have a corresponding match in DomainNet
            # note that since captions are synthetic, the class name always appears in the caption
            if any(clss in caption for clss in shared_class_names):
                continue

            sketches_filtered.append(sketch)

        random.seed(42)
        indices = random.sample(range(len(domainnet)), k=len(sketches_filtered))
        domainnet_filtered = [domainnet[i] for i in range(len(domainnet)) if i not in indices]
        assert len(samples) == len(captions) + len(domainnet_filtered) + len(sketches_filtered)

        samples = ['filepath\ttitle\n'] + captions + domainnet_filtered + sketches_filtered

        out_path = f'{args.indices_path}/{name}-with-in-sketches.tsv'
        with open(out_path, 'w') as f:
            f.writelines(samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Configure Imagenet-Sketch integration.')
    parser.add_argument(
        '--imagenet_path', type=str, required=True, help='path to imagenet directory (containing sketch)'
    )
    parser.add_argument('--indices_path', type=str, default='data/indices', help='path to indices')
    parser.add_argument(
        '--class_mapping_path', type=str, default='data/in_to_dn_mapping.json', help='path to the class mapping file'
    )

    args = parser.parse_args()
    main(args)
