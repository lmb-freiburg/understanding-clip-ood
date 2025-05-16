import typing

import open_clip
import torch
from open_clip import create_model_and_transforms
from torchvision.transforms import Compose

from xclip.utils import AbstractCLIP


class OpenCLIP(AbstractCLIP):
    """
    Light-weight wrapper around open_clip CLIP model to ensure a consistent interface
    with our other CLIP models and allow convenient loading of pre-trained models.
    """

    def __init__(self, clip: open_clip.model.CLIP) -> None:
        super().__init__()
        self.clip = clip

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return self.clip.encode_image(image, normalize=normalize)

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return self.clip.encode_text(text, normalize=normalize)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.clip.logit_scale.exp().clamp(0, 100)

    @classmethod
    def from_pretrained(
        cls, model_name: str, ckpt_path: str | None = None, **model_kwargs
    ) -> tuple['OpenCLIP', Compose, Compose]:
        model_kwargs['precision'] = model_kwargs.get('precision', 'fp16')  # make fp16 the default
        state_dict = None

        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict

            if next(iter(state_dict.items()))[0].startswith('module'):
                state_dict = {k[len('module.') :]: v for k, v in state_dict.items()}

            if 'logit_bias' in state_dict:
                model_kwargs['init_logit_bias'] = state_dict['logit_bias']

        clip, preprocess_train, preprocess_val = create_model_and_transforms(model_name, **model_kwargs)
        clip = typing.cast(open_clip.model.CLIP, clip)
        preprocess_train = typing.cast(Compose, preprocess_train)
        preprocess_val = typing.cast(Compose, preprocess_val)

        if state_dict:
            clip.load_state_dict(state_dict)

        return cls(clip), preprocess_train, preprocess_val
