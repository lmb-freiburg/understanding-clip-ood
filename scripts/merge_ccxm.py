import argparse
import os


def get_dn_train_samples(identifier):
    with open(os.path.join('data', 'indices', f'dn-captions-train-{identifier}.tsv')) as f:
        dn_samples = f.readlines()

    assert dn_samples[0] == 'filepath\ttitle\n'
    return dn_samples[1:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge CC3M/CC12M train samples with our domain mixtures.')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['cc3m', 'cc12m'],
    )
    args = parser.parse_args()

    with open(f'data/indices/{args.mode}-train.tsv') as f:
        ccxm_train = f.readlines()

    for identifier in [
        'lso-rs-nosketchclasses',
        'lso-cipqrs-nosketchclasses',
        'lso-cipqr-nosketchclasses',
        'lso-cr-noclipartclasses',
        'lso-cipqrs-noclipartclasses',
        'lso-ipqrs-noclipartclasses',
        'lso-real-only',
    ]:
        if os.path.isfile(f'data/indicies/{args.mode}-train-{identifier}.tsv'):
            continue

        dn_train = get_dn_train_samples(identifier)
        with open(f'data/indicies/{args.mode}-train-{identifier}.tsv', 'w') as f:
            f.writelines(ccxm_train + dn_train)
