import argparse
import os

import lightning.pytorch as pl
import open_clip
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader

from xclip.callbacks import CudaMemoryMonitor, CustomModelCheckpoint
from xclip.datasets import CombinedNet
from xclip.learner import ImageNetCaptionsLearner


def global_to_local_(args: argparse.Namespace) -> None:
    """
    Translate from global batch size and num workers to local, i.e., to the batch size and num workers to use per GPU.
    This function modifies its argument in-place.
    """
    # no translation needed if training on CPU
    if args.accelerator == 'cpu' or not torch.cuda.is_available():
        args.devices = args.devices if args.devices != -1 else 1
        return

    # compute world size
    devices = args.devices if args.devices != -1 else torch.cuda.device_count()
    num_nodes = args.num_nodes
    world_size = num_nodes * devices

    # divide global batch size by world size
    global_batch_size = args.batch_size
    if global_batch_size % world_size != 0:
        raise ValueError(
            'Global batch size needs to be divisible by world size but got '
            f'global_batch_size={global_batch_size} and world_size={world_size}.'
        )
    args.batch_size = global_batch_size // world_size
    args.global_batch_size = global_batch_size

    # divide global num workers by number of devices
    global_num_workers = args.num_workers if args.num_workers else 0
    if global_num_workers % devices != 0:
        raise ValueError(
            'Global num workers size needs to be divisible by number of devices but got '
            f'global_num_workers={global_num_workers} and devices={devices}.'
        )
    args.num_workers = global_num_workers // devices
    args.global_num_workers = global_num_workers


def train(args: argparse.Namespace) -> None:
    # set random seed for torch, numpy, and python rngs if given
    if args.seed:
        pl.seed_everything(args.seed, workers=True)

    # allow using tensor cores for matmul
    torch.set_float32_matmul_precision('high')

    # load datasets and prepare dataloaders
    train_dataset = CombinedNet(
        args.train_index_path,
        args.in_class_index_path,
        args.class_mapping_path,
        transform=open_clip.image_transform(image_size=224, is_train=True),  # type: ignore
    )
    val_dataset = CombinedNet(
        args.val_index_path,
        args.in_class_index_path,
        args.class_mapping_path,
        transform=open_clip.image_transform(image_size=224, is_train=False),  # type: ignore
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=True
    )
    learner = ImageNetCaptionsLearner(args.model, args.learning_rate, num_classes=1345)

    # create logger and callbacks
    log_dir = 'tensorboard_logs'
    log_dir = os.path.join(args.ws_path, log_dir) if args.ws_path else log_dir
    logger = TensorBoardLogger(save_dir=log_dir, name=args.experiment_name)

    ckpt_dir = f'checkpoints/{args.experiment_name}'
    ckpt_dir = os.path.join(args.ws_path, ckpt_dir) if args.ws_path else ckpt_dir
    callbacks = [
        CudaMemoryMonitor(),
        LearningRateMonitor(logging_interval='step'),
        CustomModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=-1,
            save_first=True,
            every_n_epochs=args.save_epochs,
            save_weights_only=True,
        ),
        CustomModelCheckpoint(dirpath=ckpt_dir, filename='last', save_top_k=1),
    ]

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        detect_anomaly=args.detect_anomaly,
    )

    trainer.fit(learner, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure CLIP training parameters.')
    parser.add_argument('experiment_name', type=str, help='the name for the current experiment')
    parser.add_argument(
        '--model',
        type=str,
        default='rn50-clip',
        choices=['vit-b-32-timm', 'vit-b-32-clip', 'rn50-clip'],
        help='the model type to train',
    )

    # learner config
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate used for training')

    # trainer config
    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=('cpu', 'gpu', 'auto'),
        help='the accelerator used for training',
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='auto',
        choices=('ddp', 'fsdp', 'auto'),
        help='parallelization strategy for the trainer',
    )
    parser.add_argument('--devices', type=int, default=-1, help='the number of devices to train on')
    parser.add_argument('--num_nodes', type=int, default=1, help='the number of gpu nodes to train on')
    parser.add_argument(
        '--precision',
        type=str,
        default='16-mixed',
        choices=('16-mixed', 'bf16-mixed', '32-true', '64-true'),
        help='the floating point precision to use',
    )
    parser.add_argument('--max_epochs', type=int, default=90, help='the maximum number of training steps')
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='value for clipping global gradient norm')
    parser.add_argument('--detect_anomaly', action='store_true', help='enables torch.autograd anomaly detection')
    parser.add_argument('--ckpt_path', type=str, default=None, help='optional checkpoint to resume training from')
    parser.add_argument('--save_epochs', type=int, default=5, help='save checkpoints every n training epochs')

    # data config
    parser.add_argument('--train_index_path', type=str, help='path to the training index file')
    parser.add_argument('--val_index_path', type=str, help='path to the validation index file')
    parser.add_argument(
        '--in_class_index_path',
        type=str,
        default='data/imagenet_class_index.json',
        help='path to the in class index file',
    )
    parser.add_argument(
        '--class_mapping_path', type=str, default='data/in_to_dn_mapping.json', help='path to the class mapping file'
    )
    parser.add_argument('--batch_size', type=int, default=256, help='the global batch size to use for training')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='the number of workers for dataloaders')
    parser.add_argument('--ws_path', type=str, help=('optional path to a workspace for storing checkpoints and logs'))
    parser.add_argument('--seed', type=int, default=0, help='random seed for lightning seed_everything')

    args = parser.parse_args()
    global_to_local_(args)
    train(args)
