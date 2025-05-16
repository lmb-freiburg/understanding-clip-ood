"""
MIT License

Copyright (c) 2024 Sukrut Rao, Sweta Mahajan, Moritz Böhle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File taken and adapted to our setup from:
https://github.com/neuroexplicit-saar/Discover-then-Name/blob/main/scripts/train_sae.py
"""

import argparse
import math
import os
import random
import shutil
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from xclip.datasets import DomainNetCaptions, TsvDataset
from xclip.open_clip import OpenCLIP
from xclip.sparse_autoencoder import Pipeline

# from dncbm.arg_parser import get_common_parser
# from dncbm.utils import common_init


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_activations_domainnet(args: argparse.Namespace) -> None:
    os.makedirs(os.path.join(args.out_dir, 'activations'), exist_ok=True)

    # load datasets and tokenizers
    clip, _, preprocess_val = OpenCLIP.from_pretrained(args.img_enc_name, ckpt_path=args.ckpt_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip.to(device)
    clip.eval()

    if not os.path.exists(os.path.join(args.out_dir, 'activations', 'train_activations.pth')):
        dataset = DomainNetCaptions(args.domainnet_path, 'train', transform=preprocess_val, mode='none')
        dataloader = DataLoader(dataset, batch_size=args.activations_bs, num_workers=args.num_workers, shuffle=True)

        activations = []
        for batch in tqdm.tqdm(dataloader, desc='Precomputing train activations'):
            with torch.inference_mode():
                activations.append(F.normalize(clip.encode_image(batch.half().to(device))))

        img_feat = torch.cat(activations).cpu()
        img_feat = img_feat[torch.randperm(img_feat.size(0))]
        torch.save(img_feat, os.path.join(args.out_dir, 'activations', 'train_activations.pth'))

        del img_feat
    else:
        print('Train activations already saved. Skipping precomputation')

    if not os.path.exists(os.path.join(args.out_dir, 'activations', 'train_val_activations.pth')):
        dataset = DomainNetCaptions(args.domainnet_path, 'val', transform=preprocess_val, mode='none')
        dataloader = DataLoader(dataset, batch_size=args.activations_bs, num_workers=args.num_workers, shuffle=True)

        img_feat = []
        for batch in tqdm.tqdm(dataloader, desc='Precomputing val activations'):
            with torch.inference_mode():
                img_feat.append(F.normalize(clip.encode_image(batch.half().to(device))))

        img_feat = torch.cat(img_feat).cpu()
        img_feat = img_feat[torch.randperm(img_feat.size(0))]
        torch.save(img_feat, os.path.join(args.out_dir, 'activations', 'train_val_activations.pth'))

        del img_feat
    else:
        print('Val activations already saved. Skipping precomputation')

    del clip
    torch.cuda.empty_cache()


def save_activations_cc12m(args: argparse.Namespace) -> None:
    os.makedirs(os.path.join(args.out_dir, 'activations'), exist_ok=True)

    # load datasets and tokenizers
    clip, _, preprocess_val = OpenCLIP.from_pretrained(args.img_enc_name, ckpt_path=args.ckpt_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip.to(device)
    clip.eval()

    if not all(
        [os.path.exists(os.path.join(args.out_dir, 'activations', f'train_activations_{i}.pth')) for i in range(32)]
    ):
        dataset = ConcatDataset(
            (
                DomainNetCaptions(args.domainnet_path, 'train', transform=preprocess_val, mode='none'),
                TsvDataset(
                    os.path.join(args.cc12m_path, 'cc12m-train.tsv'),
                    img_transform=preprocess_val,
                    return_caption=False,
                ),
            )
        )
        dataloader = DataLoader(dataset, batch_size=args.activations_bs, num_workers=args.num_workers, shuffle=True)

        i = 0
        activations = []
        for batch in tqdm.tqdm(dataloader, desc='Precomputing train activations'):
            with torch.inference_mode():
                activations.append(F.normalize(clip.encode_image(batch.half().to(device))))

            if len(activations) == 295 and i != 31:
                img_feat = torch.cat(activations).cpu()
                img_feat = img_feat[torch.randperm(img_feat.size(0))]
                torch.save(img_feat, os.path.join(args.out_dir, 'activations', f'train_activations_{i}.pth'))
                i += 1
                del img_feat
                activations = []

        img_feat = torch.cat(activations).cpu()
        img_feat = img_feat[torch.randperm(img_feat.size(0))]
        torch.save(img_feat, os.path.join(args.out_dir, 'activations', f'train_activations_{i}.pth'))

        del img_feat
    else:
        print('Train activations already saved. Skipping precomputation')

    if not os.path.exists(os.path.join(args.out_dir, 'activations', 'train_val_activations.pth')):
        dataset = ConcatDataset(
            (
                DomainNetCaptions(args.domainnet_path, 'val', transform=preprocess_val, mode='none'),
                TsvDataset(
                    os.path.join(args.cc12m_path, 'cc12m-val.tsv'),
                    img_transform=preprocess_val,
                    return_caption=False,
                ),
            )
        )
        dataloader = DataLoader(dataset, batch_size=args.activations_bs, num_workers=args.num_workers, shuffle=True)

        img_feat = []
        for batch in tqdm.tqdm(dataloader, desc='Precomputing val activations'):
            with torch.inference_mode():
                img_feat.append(F.normalize(clip.encode_image(batch.half().to(device))))

        img_feat = torch.cat(img_feat).cpu()
        img_feat = img_feat[torch.randperm(img_feat.size(0))]
        torch.save(img_feat, os.path.join(args.out_dir, 'activations', 'train_val_activations.pth'))

        del img_feat
    else:
        print('Val activations already saved. Skipping precomputation')

    del clip
    torch.cuda.empty_cache()


def save_activations(args: argparse.Namespace) -> None:
    if args.domainnet_only:
        save_activations_domainnet(args)
    else:
        save_activations_cc12m(args)


def train_sae(args: argparse.Namespace) -> None:
    start_time = time()
    shutil.rmtree(os.path.join(args.out_dir, 'checkpoints'), ignore_errors=True)
    os.makedirs(os.path.join(args.out_dir, 'checkpoints'), exist_ok=False)
    # embeddings_path = (
    #     f'/BS/language-explanations/work/concept_naming/embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth'
    # )
    # args.vocab_specific_embedding = torch.load(embeddings_path).to(args.device)

    autoencoder_input_dim: int = args.input_dim
    n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
    autoencoder = SparseAutoencoder(
        n_input_features=autoencoder_input_dim,
        n_learned_features=n_learned_features,
        n_components=len(args.hook_points),
    ).to(args.device)
    print(f'Autoencoder created at {time() - start_time} seconds')

    print(f'------------Getting Image activations from directory: {args.out_dir}/activations')
    print(f'------------Getting Image activations from model: {args.img_enc_name}')

    # We use a loss reducer, which simply adds up the losses from the underlying loss functions.
    loss = LossReducer(
        LearnedActivationsL1Loss(
            l1_coefficient=float(args.l1_coeff),
        ),
        L2ReconstructionLoss(),
    )
    print(f'Loss created at {time() - start_time} seconds')

    optimizer = AdamWithReset(
        params=autoencoder.parameters(),
        named_parameters=autoencoder.named_parameters(),
        lr=float(args.lr),
        betas=(float(args.adam_beta_1), float(args.adam_beta_2)),
        eps=float(args.adam_epsilon),
        weight_decay=float(args.adam_weight_decay),
        has_components_dim=True,
    )

    print(f'Optimizer created at {time() - start_time} seconds')
    actual_resample_interval = 1
    activation_resampler = ActivationResampler(
        resample_interval=actual_resample_interval,
        n_activations_activity_collate=actual_resample_interval,
        max_n_resamples=math.inf,  # type: ignore
        n_learned_features=n_learned_features,
        resample_epoch_freq=args.resample_freq,
        resample_dataset_size=args.resample_dataset_size,
    )

    print(f'Activation resampler created at {time() - start_time} seconds')

    if args.use_wandb:
        print('wandb started!')

        wandb_project_name = f'SAEImg_{args.img_enc_name}'

        print(f'wandb started! {wandb_project_name}')

        wandb_dir = os.path.join(args.out_dir, '.cache/')
        wandb_path = Path(wandb_dir)
        wandb_path.mkdir(exist_ok=True)
        wandb.init(
            project=wandb_project_name,
            dir=wandb_dir,
            name=args.img_enc_name,
            config=args,  # type: ignore
            entity='text_concept_explanations',
            anonymous='allow',
        )

        wandb.define_metric('custom_steps')
        wandb.define_metric('train/loss_instability_across_batches', step_metric='custom_steps')

        print(f'Wandb initialized at {time() - start_time} seconds')

    logger = SummaryWriter(log_dir=os.path.join(args.out_dir, 'tensorboard'))
    pipeline = Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        checkpoint_directory=Path(f'{args.out_dir}/checkpoints'),
        loss=loss,
        optimizer=optimizer,
        device=args.device,
        logger=logger,
        args=args,
    )
    print(f'Pipeline created at {time() - start_time} seconds')

    fnames = os.listdir(f'{args.out_dir}/activations')
    print(f'Getting fnames from {args.out_dir}/activations')

    train_fnames = []
    train_val_fnames = []
    for fname in fnames:
        if fname.startswith('train_val'):
            train_val_fnames.append(os.path.join(os.path.abspath(f'{args.out_dir}/activations'), fname))
        elif fname.startswith('train'):
            train_fnames.append(os.path.join(os.path.abspath(f'{args.out_dir}/activations'), fname))
    if args.val_freq == 0:
        train_fnames = train_fnames + train_val_fnames
        train_val_fnames = None

    print(f'Train and Train_val fnames created at {time() - start_time} seconds')

    # It takes the train activations and inside split it into train_activations and train_val_activations
    pipeline.run_pipeline(
        train_batch_size=int(args.train_sae_bs),
        checkpoint_frequency=int(args.ckpt_freq),
        val_frequency=int(args.val_freq),
        num_epochs=args.num_epochs,
        train_fnames=train_fnames,
        train_val_fnames=train_val_fnames,
        start_time=start_time,
        resample_epoch_freq=args.resample_freq,
    )

    print(f'-------total time taken------ {np.round(time() - start_time, 3)}')


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    save_activations(args)
    train_sae(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l1_coeff', type=float, default=3e-4)

    # Adam parameters (set to the default ones here)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_beta_1', type=float, default=0.9)
    parser.add_argument('--adam_beta_2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--adam_weight_decay', type=float, default=0.0)

    parser.add_argument(
        '--img_enc_name',
        type=str,
        default='RN50',
        help='Name of the clip image encoder',
        choices=['RN50'],
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Path to the directory where the checkpoints and features should be saved',
    )
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint to load the model from')
    parser.add_argument('--domainnet_path', type=str, required=True, help='Path to the domainnet dataset')
    parser.add_argument('--cc12m_path', type=str, required=True, help='Path to the domainnet dataset')
    parser.add_argument('--domainnet_only', action='store_true', default=False)
    parser.add_argument('--activations_bs', type=int, default=1024, help='batch size to precompute image activations')
    parser.add_argument('--num_workers', type=int, default=6, help='num workers for precomputing image activations')
    parser.add_argument(
        '--hook_points', nargs='*', help='Name of the model hook points to get the activations from', default=['out']
    )

    parser.add_argument('--resample_freq', type=int, default=500_000)  # 122_880_000
    parser.add_argument('--resample_dataset_size', type=int, default=819_200)
    parser.add_argument('--val_freq', type=int, default=50_000, help='number of samples after which to run validation')
    parser.add_argument(
        '--ckpt_freq', type=int, default=500_000, help='number of samples after which to save the checkpoint'
    )

    # SAE related
    parser.add_argument('--input_dim', type=int, default=1024, help='dimension of the input to the SAE')
    parser.add_argument('--train_sae_bs', type=int, default=4096, help='batch size to train SAE')
    parser.add_argument('--expansion_factor', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train the SAE')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=49)
    parser.add_argument('--save_suffix', type=str, default='')

    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_entity', type=str, default='dncbm')

    args = parser.parse_args()
    main(args)
