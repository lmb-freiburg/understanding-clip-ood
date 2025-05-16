import argparse
import os
import random

import tqdm

terms = {
    'all': ['image', 'picture'],
    'clipart': ['clipart', 'illustration'],
    'infograph': ['infograph', 'informational chart'],
    'painting': ['painting', 'art'],
    'quickdraw': ['quickdraw', 'doodle'],
    'real': ['photo', 'snapshot'],
    'sketch': ['sketch', 'drawing'],
}


aans = {
    'image': 'an ',
    'picture': 'a ',
    'clipart': 'a ',
    'illustration': 'an ',
    'infograph': 'an ',
    'informational chart': 'an ',
    'painting': 'a ',
    'art': '',
    'quickdraw': 'a ',
    'doodle': 'a ',
    'photo': 'a ',
    'snapshot': 'a ',
    'sketch': 'a ',
    'drawing': 'a ',
}


templates = [
    '{AAN}{TERM} of a {CLS}.',
    'a {CLS} {TERM}.',
    '{AAN}{TERM} depicting a {CLS}.',
    'a {CLS} depicted in {AAN}{TERM}.',
    '{AAN}{TERM} showing a {CLS}.',
    'a {CLS} is visible in {AAN}{TERM}.',
]


def insert_caption_to_sample(sample: str, exclude_domain_terms: bool) -> str:
    path, label = sample.split()
    domain, cls, *_ = path.split('/')
    cls = cls.replace('_', ' ')

    # choose random template
    template = random.choice(templates)
    assert template[-1] == '.'
    template = template if random.random() < 0.5 else template[:-1]  # randomly drop the full stop

    # choose random term
    term = random.choice(terms['all']) if exclude_domain_terms else random.choice(terms['all'] + terms[domain])
    aan = aans[term]

    return '\t'.join([path, label, template.format(CLS=cls, TERM=term, AAN=aan)])


def main(args: argparse.Namespace):
    random.seed(args.seed)
    for domain in tqdm.tqdm(
        ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'], desc='Generating captions'
    ):
        for split in ['train', 'test']:
            with open(os.path.join(args.domainnet_path, f'{domain}_{split}.txt')) as f:
                samples = f.readlines()

            samples = [insert_caption_to_sample(sample, exclude_domain_terms=False) + '\n' for sample in samples]

            with open(os.path.join(args.domainnet_path, f'{domain}_{split}.tsv'), 'w') as f:
                f.writelines(samples)

    # re-seed
    random.seed(args.seed)
    os.makedirs(os.path.join(args.domainnet_path, 'aligned-captions'), exist_ok=True)
    for domain in tqdm.tqdm(
        ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'], desc='Generating aligned captions'
    ):
        aligned_domain_path = os.path.join(args.domainnet_path, 'aligned-captions', domain)
        if not os.path.exists(aligned_domain_path):
            os.symlink(os.path.join(args.domainnet_path, domain), aligned_domain_path)

        for split in ['train', 'test']:
            with open(os.path.join(args.domainnet_path, f'{domain}_{split}.txt')) as f:
                samples = f.readlines()

            samples = [insert_caption_to_sample(sample, exclude_domain_terms=True) + '\n' for sample in samples]

            with open(os.path.join(args.domainnet_path, 'aligned-captions', f'{domain}_{split}.tsv'), 'w') as f:
                f.writelines(samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure DomainNet caption generation.')
    parser.add_argument('--domainnet_path', type=str, help='path to imagenet directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    main(args)
