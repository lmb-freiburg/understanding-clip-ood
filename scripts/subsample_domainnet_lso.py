import argparse
import math
import os

from xclip.datasets import DomainNetCaptions


def merge_files(split: str, identifier: str, indices_path: str) -> None:
    assert os.path.isfile(os.path.join(indices_path, f'in-captions-{split}.tsv')), f'Missing in-captions-{split}.tsv'

    with open(os.path.join(indices_path, f'in-captions-{split}.tsv')) as f:
        in_captions = f.readlines()
        assert in_captions[0] == 'filepath\ttitle\n'

    with open(os.path.join(indices_path, f'dn-captions-{split}-{identifier}.tsv')) as f:
        dn_captions = f.readlines()
        assert dn_captions[0] == 'filepath\ttitle\n'
        dn_captions = dn_captions[1:]

    with open(os.path.join(indices_path, f'combined-captions-{split}-{identifier}.tsv'), 'w') as f:
        f.writelines(in_captions)
        f.writelines(dn_captions)


def main(args: argparse.Namespace):
    args.indices_path = os.path.abspath(args.indices_path)
    args.domainnet_path = os.path.abspath(args.domainnet_path)

    # if not args.subsample:
    #     warnings.warn('Not subsampling is deprecated and subsample will be set to True.')
    #     args.subsample = True

    assert args.exclude is not None or args.real_only, 'Must specify either exclude or real_only'
    assert args.allow_pct == 0 or args.subsample, 'allow_pct requires subsampling'
    assert not args.pseudo_exclude or args.allow_pct == 0, 'Cannot use pseudo_exclude with allow_pct'
    assert 'aligned-captions' not in args.domainnet_path or args.aligned_captions, (
        'aligned-captions directory should only be used with aligned_captions flag'
    )

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

    if args.real_only:
        # sanity stuff to avoid accidental misconfigurations
        assert args.exclude_domains == [], 'Cannot exclude domains when using real_only'
        assert args.exclude is None, 'Cannot exclude class when using real_only'
        assert args.pseudo_exclude is False, 'Cannot use pseudo_exclude when using real_only'
        assert args.single_domain is False, 'Cannot use single_domain when using real_only'
        assert args.subsample is False, 'Cannot subsample when using real_only'

        args.exclude_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch']
        filter_classes = {}
    else:
        domain_to_exclude = args.exclude
        assert domain_to_exclude in ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch']
        filter_classes = (
            {domain_to_exclude: set(class_to_idx.values())} if not args.pseudo_exclude and args.allow_pct == 0 else {}
        )

        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        if args.single_domain:
            assert args.exclude_domains == [], 'Cannot specify both single_domain and exclude_domains'
            args.exclude_domains = [d for d in domains if d != domain_to_exclude and d != 'real']
        else:
            assert args.exclude_domains is not None, 'Must specify either single_domain or exclude_domains'
            assert 'real' not in args.exclude_domains, 'Cannot exclude real domain'
            # assert domain_to_exclude not in args.exclude_domains, 'Cannot exclude domain to exclude'
            assert all([d in domains for d in args.exclude_domains]), 'Unknown domain in exclude_domains'

    print(f'{filter_classes=}')
    print(f'{args.exclude_domains=}')

    dataset_train = DomainNetCaptions(
        args.domainnet_path,
        'train',
        transform=lambda x: x,
        exclude_domains=args.exclude_domains,
        filter_classes=filter_classes,
    )
    dataset_val = DomainNetCaptions(
        args.domainnet_path,
        'val',
        transform=lambda x: x,
        exclude_domains=args.exclude_domains,
        filter_classes=filter_classes,
    )

    # sanity check
    domainnet_classes = {}
    for path, label, _ in dataset_val.samples:
        *_, clss, _ = path.split('/')
        domainnet_classes[label] = clss.replace('_', ' ')

    for cls, label in class_to_idx.items():
        assert domainnet_classes[label] == cls, f'{domainnet_classes[label]=} {cls=}'

    if args.subsample and not args.real_only:
        print('Subsampling training set')
        domain_index = {d: {} for d in domains}
        for sample in dataset_train.samples:
            path, *_ = sample
            *_, domain, cls, _ = path.split('/')
            cls = cls.replace('_', ' ')
            domain_index[domain][cls] = domain_index[domain].get(cls, []) + [sample]
        assert len(dataset_train) == sum([sum([len(s) for s in domain_index[d].values()]) for d in domains])

        # compute reference size for subsampling as the size of the single rendition setting
        subsample_size = len(
            DomainNetCaptions(
                args.domainnet_path,
                'train',
                transform=lambda x: x,
                exclude_domains=[
                    d for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch'] if d != domain_to_exclude
                ],
                filter_classes={domain_to_exclude: set(class_to_idx.values())},
            )
        )

        # compute number of allowed samples from the excluded classes
        shrink_factor = subsample_size / len(dataset_train)

        if args.allow_pct > 0:
            allowed_cls_samples = sum(
                [
                    math.ceil(args.allow_pct * shrink_factor * len(domain_index[domain_to_exclude][cls]))
                    for cls in class_to_idx
                ]
            )
            maximum_cls_samples = sum([len(domain_index[domain_to_exclude][cls]) for cls in class_to_idx])
            assert args.allow_pct > 0 or allowed_cls_samples == 0, f'{allowed_cls_samples=}'
            pseudo_shrink_factor = shrink_factor
            shrink_factor = (subsample_size - allowed_cls_samples) / (len(dataset_train) - maximum_cls_samples)

        print(f'\t{len(dataset_train)=}')
        print(f'\t{subsample_size=}')
        print(f'\t{shrink_factor=}')

        sub_index = {}
        for domain in domains:
            sub_index[domain] = {}
            for cls in domain_index[domain]:
                if args.allow_pct > 0 and domain == domain_to_exclude and cls in class_to_idx:
                    sub_index[domain][cls] = domain_index[domain][cls][
                        : math.ceil(args.allow_pct * pseudo_shrink_factor * len(domain_index[domain][cls]))
                    ]
                else:
                    sub_index[domain][cls] = domain_index[domain][cls][
                        : math.ceil(shrink_factor * len(domain_index[domain][cls]))
                    ]

        current_size = sum([sum([len(s) for s in sub_index[d].values()]) for d in domains])
        assert current_size >= subsample_size, f'{current_size=} {subsample_size=}'
        while current_size != subsample_size:
            # start discarding from the largest domains
            for domain in ['real', 'quickdraw', 'painting', 'sketch', 'infograph', 'clipart']:
                if domain not in sub_index:
                    continue

                for cls in sub_index[domain]:
                    # do not discard any more samples from the target class if using allow_pct
                    if args.allow_pct > 0 and domain == domain_to_exclude and cls in class_to_idx:
                        continue

                    sub_index[domain][cls].pop()
                    current_size -= 1
                    if current_size == subsample_size:
                        break

                if current_size == subsample_size:
                    break

        assert subsample_size == sum([sum([len(s) for s in sub_index[d].values()]) for d in domains])
        # set dataset samples using the subsampled index
        dataset_train.samples = [
            sample for domain in domains for cls in sub_index[domain] for sample in sub_index[domain][cls]
        ]
        assert subsample_size == len(dataset_train)
        print(f'\t{len(dataset_train)=}')

    if args.real_only:
        identifier = 'real-only'
    else:
        identifier = (
            ''.join([d[0] for d in domains if d not in args.exclude_domains]) + f'-no{domain_to_exclude}classes'
        )

        if not args.subsample:
            identifier += '-nosub'

        if args.pseudo_exclude:
            identifier += '-pseudo'

        if args.allow_pct > 0:
            identifier += f'-allow{args.allow_pct}'

        if args.aligned_captions:
            identifier += '-aligned'

    identifier = f'lso-{identifier}'
    print(f'{identifier=}')

    # for safety, check if files already exist
    if not args.override:
        assert not os.path.isfile(os.path.join(args.indices_path, f'dn-captions-train-{identifier}.tsv')), (
            f'dn-captions-train-{identifier}.tsv already exists'
        )
        assert not os.path.isfile(os.path.join(args.indices_path, f'dn-captions-val-{identifier}.tsv')), (
            f'dn-captions-val-{identifier}.tsv already exists'
        )
        assert not os.path.isfile(os.path.join(args.indices_path, f'combined-captions-train-{identifier}.tsv')), (
            f'combined-captions-train-{identifier}.tsv already exists'
        )
        assert not os.path.isfile(os.path.join(args.indices_path, f'combined-captions-val-{identifier}.tsv')), (
            f'combined-captions-val-{identifier}.tsv already exists'
        )

    dataset_train.to_tsv(os.path.join(args.indices_path, f'dn-captions-train-{identifier}.tsv'))
    dataset_val.to_tsv(os.path.join(args.indices_path, f'dn-captions-val-{identifier}.tsv'))

    merge_files('train', identifier, args.indices_path)
    merge_files('val', identifier, args.indices_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure DomainNet subsampling.')
    parser.add_argument('--indices_path', type=str, required=True, default='data/indices', help='out path for indices')
    parser.add_argument('--domainnet_path', type=str, required=True, help='path to domainnet directory')
    parser.add_argument('--exclude', type=str, help='domain to exclude classes from')
    parser.add_argument('--pseudo_exclude', action='store_true', help='do not actually exclude the classes')
    parser.add_argument('--single_domain', action='store_true', help='only use specified domain and real images')
    parser.add_argument('--exclude_domains', type=str, nargs='*', default=[], help='domains to (completely) exclude')
    parser.add_argument('--subsample', action='store_true', help='keep dataset size consistent with more renditions')
    parser.add_argument(
        '--allow_pct', type=float, default=0, help='allow some percentage of samples from the excluded classes'
    )
    parser.add_argument('--aligned_captions', action='store_true', help='use captions without domain-specific keywords')
    parser.add_argument('--real_only', action='store_true', help='only use real images')
    parser.add_argument('--override', action='store_true', help='override existing files')

    args = parser.parse_args()
    main(args)
