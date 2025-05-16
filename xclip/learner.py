from typing import Any

import lightning.pytorch as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from xclip.open_clip import OpenCLIP


class ImageNetCaptionsLearner(pl.LightningModule):
    def __init__(self, model: str, lr: float, num_classes: int = 1000) -> None:
        super().__init__()

        if model == 'vit-b-32-timm':
            self.backbone = timm.models.vision_transformer.vit_base_patch32_clip_224(num_classes=num_classes)
            self.head = nn.Identity()
            self.is_vit = True
        elif model == 'vit-b-32-clip':
            # if ckpt_path is not provided, `from_pretrained` will load the model with random weights
            self.backbone = OpenCLIP.from_pretrained('ViT-B-32', precision='fp32')[0].clip.visual
            self.head = nn.Linear(512, num_classes)
            self.is_vit = True
        elif model == 'rn50-clip':
            # if ckpt_path is not provided, `from_pretrained` will load the model with random weights
            self.backbone = OpenCLIP.from_pretrained('RN50', precision='fp32')[0].clip.visual
            self.head = nn.Linear(1024, num_classes)
            self.is_vit = False
        else:
            raise ValueError(f'Invalid model: {model}')

        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.relu(x)
        x = self.head(x)
        return x

    def compute_and_log_loss(self, batch: tuple, suffix: str) -> torch.Tensor:
        """
        Handle forward pass, loss computation, and logging. Used for both training and validation steps.
        """
        imgs, labels = batch

        # compute image and text embeddings and gather outputs across GPUs in case of distributed training
        logits = self.forward(imgs)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()

        self.log(f'Loss/{suffix}', loss, sync_dist=True)
        self.log(f'Accuracy/{suffix}', acc, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch: tuple, _) -> torch.Tensor:
        assert self.training
        return self.compute_and_log_loss(batch, suffix='train')

    def validation_step(self, batch: tuple, _) -> torch.Tensor:
        return self.compute_and_log_loss(batch, suffix='valid')

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.SGD(
            self.parameter_groups(), lr=self.lr, momentum=0.9, weight_decay=0.0001, nesterov=True
        )
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[11, 18, 25], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    def parameter_groups(self) -> list[dict]:
        """Group parameters into weight decay and no weight decay."""

        def exclude(n: str, p: nn.Parameter) -> bool:
            return p.ndim < 2 or 'bn' in n or 'ln' in n or 'bias' in n or 'logit_scale' in n

        def include(n: str, p: nn.Parameter) -> bool:
            return not exclude(n, p)

        named_parameters = list(self.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        return [{'params': gain_or_bias_params, 'weight_decay': 0}, {'params': rest_params}]
