import itertools
import math
import os
import warnings
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

num_domainnet_classes = 345

domains = ['real', 'clipart', 'infograph', 'painting', 'quickdraw', 'sketch']

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


def plot(layer_domain_a_vs_domain_b, domains, save_dir, cls_subset):
    df = (
        pd.DataFrame(layer_domain_a_vs_domain_b)
        .reset_index(names=['domain_a', 'domain_b'])
        .melt(id_vars=['domain_a', 'domain_b'], var_name='layer', value_name='score')
    )
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
    # plt.title('Neuron analysis - OOD')
    plt.ylabel('Neuron sharing across domains', fontsize=11)
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
    plt.ylabel('Neuron sharing across domains', fontsize=11)
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
    plt.ylabel('Neuron sharing across domains', fontsize=11)
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
    plt.title('Neuron analysis - OOD')
    plt.ylabel('Overlap score')
    plt.xlabel('Layer')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'overlap_score_{cls_subset}_domains.pdf'))
    plt.close()


def main(args):
    circuit_analysis_dir = os.path.join(args.model_dir, 'circuit_analysis')
    assert os.path.isdir(circuit_analysis_dir), f'circuit analysis directory not found at {circuit_analysis_dir}'
    assert all([os.path.isdir(os.path.join(circuit_analysis_dir, domain)) for domain in domains]), (
        'circuit analysis directories not found for all domains'
    )

    pt_files = [f for f in os.listdir(os.path.join(circuit_analysis_dir, 'real')) if f.endswith('_nodes.pt')]
    # print(pt_files)

    all_layer_domain_a_vs_domain_b, ood_layer_domain_a_vs_domain_b, id_layer_domain_a_vs_domain_b = {}, {}, {}
    missing_files = []
    for pt_file in tqdm(pt_files, leave=False):  # iterate class ids
        for domain_a, domain_b in itertools.combinations(domains, r=2):
            # assert os.path.isfile(os.path.join(circuit_analysis_dir, domain_a, pt_file)), f'file not found at {os.path.join(circuit_analysis_dir, domain_a, pt_file)}'
            if not os.path.isfile(os.path.join(circuit_analysis_dir, domain_a, pt_file)):
                if os.path.join(circuit_analysis_dir, domain_a, pt_file) not in missing_files:
                    print(f'file not found at {os.path.join(circuit_analysis_dir, domain_a, pt_file)}')
                missing_files.append(os.path.join(circuit_analysis_dir, domain_a, pt_file))
                continue
            if not os.path.isfile(os.path.join(circuit_analysis_dir, domain_b, pt_file)):
                if os.path.join(circuit_analysis_dir, domain_b, pt_file) not in missing_files:
                    print(f'file not found at {os.path.join(circuit_analysis_dir, domain_b, pt_file)}')
                missing_files.append(os.path.join(circuit_analysis_dir, domain_b, pt_file))
                continue
            nodes_a = torch.load(os.path.join(circuit_analysis_dir, domain_a, pt_file), map_location='cpu')
            nodes_b = torch.load(os.path.join(circuit_analysis_dir, domain_b, pt_file), map_location='cpu')
            for layer_name in nodes_a.keys():
                if layer_name == 'input':
                    continue
                num_neurons = nodes_a[layer_name].act.size(0)
                number = math.ceil(num_neurons * 0.1)
                most_important_neurons_a = nodes_a[layer_name].act.abs().sort().indices[-number:].tolist()
                most_important_neurons_b = nodes_b[layer_name].act.abs().sort().indices[-number:].tolist()
                # score = len(set(most_important_neurons_a).intersection(most_important_neurons_b)) / number
                score = len(set(most_important_neurons_a).intersection(most_important_neurons_b)) / len(
                    set(most_important_neurons_a).union(most_important_neurons_b)
                )

                if layer_name not in all_layer_domain_a_vs_domain_b:
                    all_layer_domain_a_vs_domain_b[layer_name] = {}
                if (domain_a, domain_b) not in all_layer_domain_a_vs_domain_b[layer_name]:
                    all_layer_domain_a_vs_domain_b[layer_name][(domain_a, domain_b)] = []
                all_layer_domain_a_vs_domain_b[layer_name][(domain_a, domain_b)].append(score)

                label = int(pt_file.split('_')[0])
                if label in ood_classes.values():
                    if layer_name not in ood_layer_domain_a_vs_domain_b:
                        ood_layer_domain_a_vs_domain_b[layer_name] = {}
                    if (domain_a, domain_b) not in ood_layer_domain_a_vs_domain_b[layer_name]:
                        ood_layer_domain_a_vs_domain_b[layer_name][(domain_a, domain_b)] = []
                    ood_layer_domain_a_vs_domain_b[layer_name][(domain_a, domain_b)].append(score)
                else:  # id
                    if layer_name not in id_layer_domain_a_vs_domain_b:
                        id_layer_domain_a_vs_domain_b[layer_name] = {}
                    if (domain_a, domain_b) not in id_layer_domain_a_vs_domain_b[layer_name]:
                        id_layer_domain_a_vs_domain_b[layer_name][(domain_a, domain_b)] = []
                    id_layer_domain_a_vs_domain_b[layer_name][(domain_a, domain_b)].append(score)

    all_layer_domain_a_vs_domain_b = {
        outer_key: {
            inner_key: np.mean(inner_value) if isinstance(inner_value, list) else inner_value
            for inner_key, inner_value in outer_value.items()
        }
        for outer_key, outer_value in all_layer_domain_a_vs_domain_b.items()
    }
    plot(all_layer_domain_a_vs_domain_b, domains, circuit_analysis_dir, 'all')

    ood_layer_domain_a_vs_domain_b = {
        outer_key: {
            inner_key: np.mean(inner_value) if isinstance(inner_value, list) else inner_value
            for inner_key, inner_value in outer_value.items()
        }
        for outer_key, outer_value in ood_layer_domain_a_vs_domain_b.items()
    }
    plot(ood_layer_domain_a_vs_domain_b, domains, circuit_analysis_dir, 'ood')

    id_layer_domain_a_vs_domain_b = {
        outer_key: {
            inner_key: np.mean(inner_value) if isinstance(inner_value, list) else inner_value
            for inner_key, inner_value in outer_value.items()
        }
        for outer_key, outer_value in id_layer_domain_a_vs_domain_b.items()
    }
    plot(id_layer_domain_a_vs_domain_b, domains, circuit_analysis_dir, 'id')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Configure CLIP models for neuron analysis.')
    parser.add_argument('--model_dir', type=str, required=True, help='path to model directory')
    args = parser.parse_args()
    main(args)
