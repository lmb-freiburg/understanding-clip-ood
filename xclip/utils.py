from abc import ABC, abstractmethod
from typing import TypeVar

import torch
import torch.nn as nn
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer


class AbstractCLIP(nn.Module, ABC):
    @abstractmethod
    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        raise NotImplementedError(
            f'Subclasses of {self.__class__.__name__} need to implement their own encode_image method.'
        )

    @abstractmethod
    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        raise NotImplementedError(
            f'Subclasses of {self.__class__.__name__} need to implement their own encode_text method.'
        )

    @property
    @abstractmethod
    def logit_scale(self) -> torch.Tensor:
        raise NotImplementedError(
            f'Subclasses of {self.__class__.__name__} need to implement their own logit_scale property.'
        )

    @property
    def uses_one_hot_encoding(self) -> bool:
        return False


class TokenizerBase:
    def __call__(self, text: str | list[str]) -> torch.Tensor:
        _ = text
        raise NotImplementedError


OpenCLIPTokenizer = HFTokenizer | SimpleTokenizer
AbstractTokenizer = TokenizerBase | OpenCLIPTokenizer


T = TypeVar('T')


def identity(x: T) -> T:
    return x
