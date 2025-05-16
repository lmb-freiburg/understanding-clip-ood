import argparse
import itertools
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tueplots import bundles

from xclip.datasets import DomainNetCaptions
from xclip.open_clip import OpenCLIP

activations = {}
batch_size = 10000
max_num_samples = 10000

plt.rcParams.update(bundles.icml2024())


def plot(layer_domain_a_vs_domain_b, domains, save_dir, cls_subset):
    df = (
        pd.DataFrame(layer_domain_a_vs_domain_b)
        .reset_index(names=['domain_a', 'domain_b'])
        .melt(id_vars=['domain_a', 'domain_b'], var_name='layer', value_name='score')
    )
    df = df[df['domain_a'] != df['domain_b']]  # remove same domain comparisons
    df['domain_a_domain_b'] = df['domain_a'] + ' vs. ' + df['domain_b']
    df['layer'] = df['layer'].apply(
        lambda x: x.replace('resblock', 'resb')
        .replace('act', 'act')
        .replace('avgpool', 'avgp')
        .replace('attnpool', 'attnp')
    )
    df['quickdraw_or_other'] = df.apply(
        lambda row: 'Quickdraw' if 'quickdraw' in (row['domain_a'], row['domain_b']) else 'Other', axis=1
    )

    plt.figure(figsize=(4, 3))
    sns.lineplot(
        data=df, x='layer', y='score', hue='domain_a_domain_b', style='domain_a_domain_b', palette='muted', markers=True
    )
    sns.despine()
    plt.xticks(rotation=45)
    plt.ylabel('Repr sim', fontsize=11)
    plt.xlabel('Layer', fontsize=11)
    plt.gca().yaxis.set_tick_params(labelsize=9)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 0.99), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'overlap_score_{cls_subset}.pdf'))
    plt.close()

    plt.figure(figsize=(4, 3))
    sns.lineplot(
        data=df,
        x='layer',
        y='score',
        hue='quickdraw_or_other',
        style='quickdraw_or_other',
        palette='muted',
        markers=True,
    )
    sns.despine()
    plt.xticks(rotation=45)
    # plt.title('Neuron analysis - OOD')
    plt.ylabel('Repr sim', fontsize=11)
    plt.xlabel('Layer', fontsize=11)
    plt.gca().yaxis.set_tick_params(labelsize=9)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 0.99), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'overlap_score_{cls_subset}_quickdraw.pdf'))
    plt.close()

    plt.figure(figsize=(4, 3))
    sns.lineplot(
        data=df,
        x='layer',
        y='score',
        hue='quickdraw_or_other',
        style='quickdraw_or_other',
        palette='muted',
        markers=True,
    )
    sns.despine()
    plt.xticks(rotation=45)
    # plt.title('Neuron analysis - OOD')
    plt.ylabel('Repr sim', fontsize=11)
    plt.xlabel('Layer', fontsize=11)
    plt.gca().yaxis.set_tick_params(labelsize=9)
    plt.xticks([])
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 0.99), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'overlap_score_{cls_subset}_quickdraw_no_xticks.pdf'))
    plt.close()

    linestyles = ['-', '--', '-.', ':', (0, (1, 10)), (0, (5, 10))]
    style_cycle = cycle(linestyles)
    for domain in domains:
        linestyle = next(style_cycle)
        sns.lineplot(
            data=pd.concat([df[df['domain_a'] == domain], df[df['domain_b'] == domain]]),
            x='layer',
            y='score',
            markers=True,
            label=domain,
            linestyle=linestyle,
            palette='muted',
        )
    plt.xticks(rotation=45)
    plt.ylabel('Repr sim')
    plt.xlabel('Layer')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'overlap_score_{cls_subset}_domains.pdf'))
    plt.close()


def save_activation(name):
    def hook(module, input, output):
        if output.dim() == 4:
            output = output.mean(dim=[2, 3])
        activations[name] = output.cpu()

    return hook


@torch.inference_mode()
def get_data(
    model: OpenCLIP,
    dataset: Dataset,
    activation_dir: str,
    domain_name: str,
    num_workers: int = 8,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    save_activations, save_labels = {}, []
    for batch in tqdm(loader, leave=False, total=len(loader)):
        _ = model.encode_image(batch[0].half().to(device))

        for k, v in activations.items():
            if k not in save_activations:
                save_activations[k] = []
            save_activations[k].append(v)
        save_labels.append(batch[1])

    for k in save_activations:
        torch.save(torch.cat(save_activations[k], dim=0), os.path.join(activation_dir, f'{domain_name}_{k}.pt'))
    torch.save(torch.cat(save_labels, dim=0), os.path.join(activation_dir, f'{domain_name}_labels.pt'))


@torch.inference_mode()
def rbf(X, sigma=None):
    GX = X @ X.T
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


@torch.inference_mode()
def hsic0(K, L):
    n = K.shape[0]
    H = torch.eye(n) - torch.ones(n, n) / n
    H = H.to(K.device).to(K.dtype)
    return torch.trace(K @ H @ L @ H) / (n - 1) ** 2


@torch.inference_mode()
def hsic1(K, L):
    # unbiased estimator from https://jmlr.csail.mit.edu/papers/v13/song12a.html
    n = K.shape[0]
    assert K.shape == L.shape, f'K and L must have the same shape, got {K.shape} and {L.shape}'

    K_tilde = K.clone()
    L_tilde = L.clone()

    K_tilde.fill_diagonal_(0)
    L_tilde.fill_diagonal_(0)

    # compute terms
    trace_term = torch.trace(K_tilde @ L_tilde)
    constant_term = (1 / ((n - 1) * (n - 2))) * torch.sum(K_tilde) * torch.sum(L_tilde)
    cross_term = (2 / (n - 2)) * torch.sum(K_tilde @ L_tilde)
    return (1 / (n * (n - 3))) * (trace_term + constant_term - cross_term)


@torch.inference_mode()
def cka(K, L, hsic=hsic1):
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    return hsic_kl / torch.sqrt(hsic_kk * hsic_ll)


@torch.inference_mode()
def linear_cka(X, Y, hsic):
    X = X.double()
    Y = Y.double()

    K = X @ X.T
    L = Y @ Y.T
    return cka(K, L, hsic)


@torch.inference_mode()
def kernel_cka(X, Y, hsic, sigma=None):
    X = X.double()
    Y = Y.double()

    K = rbf(X.double(), sigma)
    L = rbf(Y.double(), sigma)
    return cka(K, L, hsic)


def main(args: argparse.Namespace) -> None:
    ckpt_filepath = os.path.join(args.model_dir, 'checkpoints/epoch_32.pt')
    assert os.path.isfile(ckpt_filepath), f'Checkpoint file not found: {ckpt_filepath}'

    model, _, preprocess_val = OpenCLIP.from_pretrained(args.model, ckpt_path=ckpt_filepath)

    # Register hooks
    model.clip.visual.act1.register_forward_hook(save_activation('act1'))
    model.clip.visual.act2.register_forward_hook(save_activation('act2'))
    model.clip.visual.act3.register_forward_hook(save_activation('act3'))
    model.clip.visual.avgpool.register_forward_hook(save_activation('avgpool'))
    activation_keys = ['act1', 'act2', 'act3', 'avgpool']
    offset = 1
    for i in range(len(model.clip.visual.layer1)):
        model.clip.visual.layer1[i].register_forward_hook(save_activation(f'resblock{i + offset}'))
        activation_keys.append(f'resblock{i + offset}')
    offset += len(model.clip.visual.layer1)
    for i in range(len(model.clip.visual.layer2)):
        model.clip.visual.layer2[i].register_forward_hook(save_activation(f'resblock{i + offset}'))
        activation_keys.append(f'resblock{i + offset}')
    offset += len(model.clip.visual.layer2)
    for i in range(len(model.clip.visual.layer3)):
        model.clip.visual.layer3[i].register_forward_hook(save_activation(f'resblock{i + offset}'))
        activation_keys.append(f'resblock{i + offset}')
    offset += len(model.clip.visual.layer3)
    for i in range(len(model.clip.visual.layer4)):
        model.clip.visual.layer4[i].register_forward_hook(save_activation(f'resblock{i + offset}'))
        activation_keys.append(f'resblock{i + offset}')
    model.clip.visual.attnpool.register_forward_hook(save_activation('attnpool'))
    activation_keys.append('attnpool')

    # Load DomainNet dataset
    ood_classes = {
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
    ood_class_indices = [ood_classes[c] for c in ood_classes]
    data = {
        'real': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'infograph', 'painting', 'quickdraw', 'sketch'],
        ),
        'quickdraw': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'infograph', 'painting', 'real', 'sketch'],
        ),
        'sketch': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'infograph', 'painting', 'real', 'quickdraw'],
        ),
        'clipart': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['infograph', 'painting', 'real', 'quickdraw', 'sketch'],
        ),
        'infograph': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'painting', 'real', 'quickdraw', 'sketch'],
        ),
        'painting': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'infograph', 'real', 'quickdraw', 'sketch'],
        ),
    }

    # get activations
    activation_dir = os.path.join(args.model_dir, 'activations')
    os.makedirs(activation_dir, exist_ok=True)
    for domain, d in data.items():
        if not args.acts_regenerate and any(f'{domain}_' in f for f in os.listdir(activation_dir)):
            continue
        get_data(model, d, activation_dir, domain_name=domain, batch_size=args.batch_size, num_workers=args.num_workers)

    # compute CKA metrics
    out_dir = os.path.join(args.model_dir, 'rsa')
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    labels = {
        domain: torch.load(os.path.join(activation_dir, f'{domain}_labels.pt')).to(device) for domain in data.keys()
    }

    for subset in ['all', 'ood', 'id']:
        for measure_type in ['linear_cka_hsic1_mean', 'kernel_cka_hsic1_mean']:
            if os.path.isfile(os.path.join(out_dir, f'{measure_type}_{subset}.pt')) and not args.scores_regenerate:
                all_layer_domain_a_vs_domain_b = torch.load(
                    os.path.join(out_dir, f'{measure_type}_{subset}.pt'), map_location='cpu'
                )
            else:
                all_layer_domain_a_vs_domain_b = {}
                for domain_a, domain_b in tqdm(
                    itertools.combinations(data.keys(), 2),
                    leave=False,
                    total=len(data.keys()) * (len(data.keys()) - 1) // 2,
                    desc=f'{measure_type} {subset}',
                ):
                    for key in tqdm(activation_keys, leave=False, desc='layers'):
                        act_domain_a = torch.load(os.path.join(activation_dir, f'{domain_a}_{key}.pt')).to(device)
                        act_domain_b = torch.load(os.path.join(activation_dir, f'{domain_b}_{key}.pt')).to(device)

                        cross_domain_per_class_vals = []
                        self_domain_per_class_vals = []
                        if subset == 'ood':
                            class_indices = torch.tensor(ood_class_indices)
                        elif subset == 'id':
                            class_indices = torch.tensor(
                                [
                                    label_idx
                                    for label_idx in torch.unique(labels[domain]).tolist()
                                    if label_idx not in ood_class_indices
                                ]
                            )
                        else:
                            class_indices = torch.unique(labels[domain])

                        class_mean_acts_a, class_mean_acts_b = [], []
                        for class_idx in class_indices:
                            class_mean_acts_a.append(act_domain_a[labels[domain_a] == class_idx].float().mean(dim=0))
                            class_mean_acts_b.append(act_domain_b[labels[domain_b] == class_idx].float().mean(dim=0))
                        class_mean_acts_a = torch.stack(class_mean_acts_a)
                        class_mean_acts_b = torch.stack(class_mean_acts_b)
                        if 'linear_cka' in measure_type:
                            val = linear_cka(class_mean_acts_a, class_mean_acts_b, hsic=hsic1)
                        elif 'kernel_cka' in measure_type:
                            val = kernel_cka(class_mean_acts_a, class_mean_acts_b, hsic=hsic1)
                        else:
                            raise NotImplementedError(f'CKA type {measure_type} not implemented.')
                        self_val_a = None
                        self_val_b = None

                        if key not in all_layer_domain_a_vs_domain_b:
                            all_layer_domain_a_vs_domain_b[key] = {}
                        if len(cross_domain_per_class_vals) > 0:
                            all_layer_domain_a_vs_domain_b[key][(domain_a, domain_b)] = np.mean(
                                cross_domain_per_class_vals
                            )
                        else:
                            all_layer_domain_a_vs_domain_b[key][(domain_a, domain_b)] = val.item()
                        if (
                            self_val_a is not None
                            and self_val_b is not None
                            and (domain_a, domain_a) not in all_layer_domain_a_vs_domain_b[key]
                        ):
                            all_layer_domain_a_vs_domain_b[key][(domain_a, domain_a)] = np.mean(
                                [val[0] for val in self_domain_per_class_vals]
                            )
                        if (
                            self_val_a is not None
                            and self_val_b is not None
                            and (domain_b, domain_b) not in all_layer_domain_a_vs_domain_b[key]
                        ):
                            all_layer_domain_a_vs_domain_b[key][(domain_b, domain_b)] = np.mean(
                                [val[1] for val in self_domain_per_class_vals]
                            )

                torch.save(all_layer_domain_a_vs_domain_b, os.path.join(out_dir, f'{measure_type}_{subset}.pt'))

            plot(all_layer_domain_a_vs_domain_b, data.keys(), out_dir, f'{measure_type}_{subset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure CLIP models to evaluate.')
    parser.add_argument('--model', type=str, required=True, help='CLIP model type')
    parser.add_argument('--model_dir', type=str, required=True, help='path to model directory')
    parser.add_argument('--domainnet_path', type=str, required=True, help='path to domainnet directory')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--device', type=str, default='cuda', help='device to run on')

    parser.add_argument('--acts_regenerate', action='store_true', help='regenerate activations')
    parser.add_argument('--scores_regenerate', action='store_true', help='regenerate computed scores')

    args = parser.parse_args()
    main(args)
