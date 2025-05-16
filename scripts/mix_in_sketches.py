import argparse
import json
import os
import random

from xclip.datasets import openai_imagenet_classes


def main(args: argparse.Namespace):
    # Create data mixtures with DomainNet and ImageNet sketches mixed in
    # All IN sketches that don't overlap are with DN classes are included
    # Only a percentage of DN sketches is kept (indicated in the name of the output file)
    name = 'combined-captions-train-lso-cipqrs-nosketchclasses'
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

    with open(args.class_mapping_path, 'r') as f:
        in_to_dn = json.load(f)

    shared_classes = []
    for clss in in_to_dn.values():
        if clss is not None:
            shared_classes.extend(clss)
    shared_class_names = [class_labels[c] for c in shared_classes]

    in_sketches_filtered = []
    for sketch in sketches:
        path, caption = sketch.split('\t')

        if any(clss in caption for clss in shared_class_names):
            continue

        in_sketches_filtered.append(sketch)

    def is_sketch(sample):
        return 'sketch' in sample.split('\t')[0]

    domainnet_sketches = [sample for sample in domainnet if is_sketch(sample)]
    domainnet_other = [sample for sample in domainnet if not is_sketch(sample)]

    sketches = {}
    for sample in domainnet_sketches:
        *_, domain, clss, path = sample.split('\t')[0].split('/')
        sketches[clss] = sketches.get(clss, []) + [sample]
    print(len(sketches))

    random.seed(42)
    dn_classes = list(sketches.keys())
    assert len(dn_classes) == 330
    half_classes = random.sample(dn_classes, k=165)
    quart_classes = random.sample(half_classes, k=82)
    thirtyp_classes = quart_classes + random.sample(list(set(half_classes).difference(quart_classes)), k=17)
    tenp_classes = random.sample(quart_classes, k=33)
    fivep_classes = random.sample(tenp_classes, k=16)
    onep_classes = random.sample(fivep_classes, k=3)
    one_class = random.sample(onep_classes, k=1)

    for mode, remaining_classes in [
        ('fiftyp', half_classes),
        ('twentyfivep', quart_classes),
        ('thirtyp', thirtyp_classes),
        ('tenp', tenp_classes),
        ('fivep', fivep_classes),
        ('onep', onep_classes),
        ('one', one_class),
    ]:
        print(f'{mode}')

        dn_sketches_remaining = []
        for clss in remaining_classes:
            dn_sketches_remaining.extend(sketches[clss])

        removed_sketches = len(domainnet_sketches) - len(dn_sketches_remaining)
        assert removed_sketches <= len(in_sketches_filtered), f'{removed_sketches=}, {len(in_sketches_filtered)=}'

        num_samples_to_drop = len(in_sketches_filtered) - removed_sketches
        indices = random.sample(range(len(domainnet_other)), k=num_samples_to_drop)
        domainnet_filtered = [domainnet_other[i] for i in range(len(domainnet_other)) if i not in indices]
        print(
            f'{len(samples)=}, {len(captions)=}, {len(domainnet_filtered)=}, {len(dn_sketches_remaining)=}, {len(in_sketches_filtered)=}'
        )
        assert len(samples) == len(captions) + len(domainnet_filtered) + len(dn_sketches_remaining) + len(
            in_sketches_filtered
        )

        samples_mixed = (
            ['filepath\ttitle\n'] + captions + domainnet_filtered + dn_sketches_remaining + in_sketches_filtered
        )

        out_path = f'{args.indices_path}/{name}-with-in-sketches-{mode}.tsv'
        with open(out_path, 'w') as f:
            f.writelines(samples_mixed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Configure Imagenet-Sketch integration.')
    parser.add_argument(
        '--imagenet_path', type=str, required=True, help='path to imagenet directory (containing sketch)'
    )
    parser.add_argument('--indices_path', type=str, default='data/indices', help='path to indices')
    parser.add_argument(
        '--in_class_index_path',
        type=str,
        default='data/imagenet_class_index.json',
        help='path to the in class index file',
    )
    parser.add_argument(
        '--class_mapping_path', type=str, default='data/in_to_dn_mapping.json', help='path to the class mapping file'
    )

    args = parser.parse_args()
    main(args)
