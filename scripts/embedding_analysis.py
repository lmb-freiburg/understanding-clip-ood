import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import umap


def main(args: argparse.Namespace) -> None:
    # extract step count from checkpoint filename
    def epoch_or_step_from_ckpt_file(filename: str) -> int:
        filename = os.path.basename(filename)
        begin = filename.find('step') + 5 if 'step' in filename else filename.find('epoch') + 6
        end = filename.find('.')
        return int(filename[begin:end])

    # retrieve and sort checkpoint files
    ckpt_files = args.ckpt_files
    ckpt_files = sorted(ckpt_files, key=epoch_or_step_from_ckpt_file)
    steps = [epoch_or_step_from_ckpt_file(file) for file in ckpt_files]

    out_dir = os.path.join(args.out_path, 'embedding_analysis')
    os.makedirs(out_dir, exist_ok=True)

    steps = [-1]

    for step, ckpt_file in tqdm.tqdm(list(zip(steps, ckpt_files)), desc='Evaluating models'):
        # assert step == epoch_or_step_from_ckpt_file(ckpt_file)

        # load data
        img_feats = np.load(os.path.join(args.out_path, 'features', 'img_feat.npy'))[step]
        domain_labels = np.load(os.path.join(args.out_path, 'features', 'domain_ids.npy'))
        cls_labels = np.load(os.path.join(args.out_path, 'features', 'domain_labels.npy'))

        # for each domain get 1000 indices randomly
        subsampled_indices = []
        subsampled_indices_wo_quickdraw = []
        for domain in np.unique(domain_labels):
            domain_indices = np.where(domain_labels == domain)[0]
            random_indices = np.random.choice(domain_indices, min(2000, len(domain_indices)), replace=False)
            subsampled_indices.extend(random_indices)
            if domain != 3:
                subsampled_indices_wo_quickdraw.extend(random_indices)

        if args.all or args.umap:
            reducer = umap.UMAP(n_components=2, densmap=True, random_state=42, verbose=True)
            embedding = reducer.fit_transform(img_feats[subsampled_indices])
            plt.scatter(embedding[:, 0], embedding[:, 1], c=domain_labels[subsampled_indices], cmap='Spectral', s=3)
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar()
            plt.savefig(os.path.join(out_dir, f'epoch_{step}_umap.png'))
            plt.close()

            reducer = umap.UMAP(n_components=2, densmap=True, random_state=42, verbose=True)
            embedding = reducer.fit_transform(img_feats[subsampled_indices_wo_quickdraw])
            plt.scatter(
                embedding[:, 0], embedding[:, 1], c=domain_labels[subsampled_indices_wo_quickdraw], cmap='Spectral', s=3
            )
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar()
            plt.savefig(os.path.join(out_dir, f'epoch_{step}_umap_wo_q.png'))
            plt.close()

        if args.all or args.heatmap:
            distances = np.empty((len(np.unique(domain_labels)), len(np.unique(domain_labels))))
            for domain_a in np.unique(domain_labels):
                for domain_b in np.unique(domain_labels):
                    if domain_a > domain_b:
                        continue
                    domain_a_indices = np.where(domain_labels == domain_a)[0]
                    domain_b_indices = np.where(domain_labels == domain_b)[0]
                    domain_a_feats = img_feats[domain_a_indices]
                    domain_b_feats = img_feats[domain_b_indices]
                    distances[domain_a, domain_b] = np.linalg.norm(
                        domain_a_feats.mean(axis=0) - domain_b_feats.mean(axis=0)
                    )
                    distances[domain_b, domain_a] = distances[domain_a, domain_b]

            sns.heatmap(distances, cmap='viridis', square=True, annot=True, cbar=True)
            plt.title('L2M distances between domains')
            plt.savefig(os.path.join(out_dir, f'epoch_{step}_heatmap_cls_mean.png'))
            plt.close()

            distances = np.empty((len(np.unique(domain_labels)), len(np.unique(domain_labels))))
            for domain_a in np.unique(domain_labels):
                for domain_b in np.unique(domain_labels):
                    if domain_a > domain_b:
                        continue
                    if domain_a == domain_b:
                        distances[domain_a, domain_b] = 0
                        continue
                    l2s = []
                    for cls_label in np.unique(cls_labels):
                        domain_a_indices = np.where(np.logical_and(domain_labels == domain_a, cls_labels == cls_label))[
                            0
                        ]
                        domain_b_indices = np.where(np.logical_and(domain_labels == domain_b, cls_labels == cls_label))[
                            0
                        ]
                        if len(domain_a_indices) == 0 or len(domain_b_indices) == 0:
                            continue
                        domain_a_feats = img_feats[domain_a_indices]
                        domain_b_feats = img_feats[domain_b_indices]
                        l2s.append(np.linalg.norm(domain_a_feats.mean(axis=0) - domain_b_feats.mean(axis=0)))
                    distances[domain_a, domain_b] = np.mean(l2s)
                    distances[domain_b, domain_a] = distances[domain_a, domain_b]

            sns.heatmap(distances, cmap='viridis', square=True, annot=True, cbar=True)
            plt.title('L2M cls-sensitive distances between domains')
            plt.savefig(os.path.join(out_dir, f'epoch_{step}_heatmap_cls_sensitive.png'))
            plt.close()

        if args.all or args.diff_plot:
            domain_not_quickdraw_indices = np.where(domain_labels != 3)[0]
            domain_quickdraw_indices = np.where(domain_labels == 3)[0]
            domain_a_feats = img_feats[domain_not_quickdraw_indices].mean(axis=0)
            domain_b_feats = img_feats[domain_quickdraw_indices].mean(axis=0)
            domain_abs_diff = np.abs(domain_a_feats - domain_b_feats)
            plt.plot(np.sort(domain_abs_diff))
            plt.xlabel('Emb index (sprted)')
            plt.ylabel('Abs diff')
            plt.title('Abs diff between quickdraw and non-quickdraw')
            plt.savefig(os.path.join(out_dir, f'epoch_{step}_quickdraw_abs_diff.png'))
            plt.close()

            for domain in np.unique(domain_labels):
                domain_not_quickdraw_indices = np.where(domain_labels != domain)[0]
                domain_quickdraw_indices = np.where(domain_labels == domain)[0]
                domain_a_feats = img_feats[domain_not_quickdraw_indices].mean(axis=0)
                domain_b_feats = img_feats[domain_quickdraw_indices].mean(axis=0)
                domain_abs_diff = np.abs(domain_a_feats - domain_b_feats)
                plt.plot(np.sort(domain_abs_diff), label=f'Domain {domain}')
            plt.xlabel('Emb index (sprted)')
            plt.ylabel('Abs diff')
            plt.legend()
            plt.title('Abs diff between domain vs. all other')
            plt.savefig(os.path.join(out_dir, f'epoch_{step}_all_abs_diff.png'))
            plt.close()

        if args.all or args.scatter:
            for domain in tqdm.tqdm(np.unique(domain_labels), leave=False):
                domain_not_domain_indices = np.where(domain_labels != domain)[0]
                domain_domain_indices = np.where(domain_labels == domain)[0]
                domain_a_feats = img_feats[domain_not_domain_indices].mean(axis=0)
                domain_b_feats = img_feats[domain_domain_indices].mean(axis=0)
                domain_abs_diff = np.abs(domain_a_feats - domain_b_feats)

                idx1 = np.argsort(domain_abs_diff)[-1]
                idx2 = np.argsort(domain_abs_diff)[-2]

                plot_df = pd.DataFrame(
                    {
                        f'dim {idx1}': np.concatenate(
                            [img_feats[domain_not_domain_indices][:, idx1], img_feats[domain_domain_indices][:, idx1]]
                        ),
                        f'dim {idx2}': np.concatenate(
                            [img_feats[domain_not_domain_indices][:, idx2], img_feats[domain_domain_indices][:, idx2]]
                        ),
                        'domain': [f'not {domain}'] * len(domain_not_domain_indices)
                        + [f'{domain}'] * len(domain_domain_indices),
                    }
                )
                sns.scatterplot(data=plot_df, x=f'dim {idx1}', y=f'dim {idx2}', hue='domain', s=1)
                plt.savefig(os.path.join(out_dir, f'epoch_{step}_scatter_domain_{domain}.png'))
                plt.close()

                plot_df = pd.DataFrame(
                    {
                        f'dim {idx1}': np.concatenate(
                            [img_feats[domain_not_domain_indices][:, idx1], img_feats[domain_domain_indices][:, idx1]]
                        )[::-1],
                        f'dim {idx2}': np.concatenate(
                            [img_feats[domain_not_domain_indices][:, idx2], img_feats[domain_domain_indices][:, idx2]]
                        )[::-1],
                        'domain': [f'{domain}'] * len(domain_domain_indices)
                        + [f'not {domain}'] * len(domain_not_domain_indices),
                    }
                )
                sns.scatterplot(data=plot_df, x=f'dim {idx1}', y=f'dim {idx2}', hue='domain', s=1)
                plt.savefig(os.path.join(out_dir, f'epoch_{step}_scatter_flipped_domain_{domain}.png'))
                plt.close()

        if args.all or args.eval:
            preds = np.load(os.path.join(args.out_path, 'domainnet', 'domain_pred.npy'))[step]
            labels = np.load(os.path.join(args.out_path, 'domainnet', 'domain_labels.npy'))
            print('DomainNet accuracy: ', (preds == labels).mean() * 100)

            preds = np.load(os.path.join(args.out_path, 'domainnet', 'real_kw_preds.npy'))[step]
            labels = np.load(os.path.join(args.out_path, 'domainnet', 'real_kw_labels.npy'))
            print('Real accuracy: ', (preds == labels).mean() * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure CLIP models to evaluate.')
    parser.add_argument('--model', type=str, required=True, help='CLIP model type')
    parser.add_argument('--ckpt_files', type=str, nargs='+', help='checkpoint to evaluate')
    parser.add_argument('--out_path', type=str, required=True, help='output directory for results.json')
    parser.add_argument('--domainnet_path', type=str, required=True, help='path to domainnet directory')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for dataloader')

    parser.add_argument('--all', action='store_true', help='run all analysis')
    parser.add_argument('--umap', action='store_true', help='whether to plot umap')
    parser.add_argument('--heatmap', action='store_true', help='whether to plot heatmap')
    parser.add_argument('--diff_plot', action='store_true', help='whether to plot diff plot')
    parser.add_argument('--scatter', action='store_true', help='whether to plot scatter plot')
    parser.add_argument('--eval', action='store_true', help='whether to evaluate model')

    args = parser.parse_args()
    main(args)
