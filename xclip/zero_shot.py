from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn.functional as F
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer

from xclip.utils import AbstractCLIP, identity


class AbstractZeroShotClassifier(ABC):
    def __init__(self, clip: AbstractCLIP, prompts: torch.Tensor) -> None:
        # zero-shot model
        self.clip = clip
        self.clip.eval()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip.to(self.device)

        # the CLIP variant from m3di uses one hot encoding for text
        self.prompts = F.one_hot(prompts, self.clip.vocab_size).float() if self.clip.uses_one_hot_encoding else prompts
        num_text_dims = 2 if self.clip.uses_one_hot_encoding else 1

        # reshape prompts to compute embeddings
        feature_shapes = self.prompts.shape[:-num_text_dims]
        input_ids = self.prompts.reshape(feature_shapes.numel(), *self.prompts.shape[-num_text_dims:])

        with torch.inference_mode():
            # compute text features for all prompts
            txt_feat = self.clip.encode_text(input_ids.to(self.device))
            assert txt_feat.ndim == 2

            # normalize text features to compute similarities later
            txt_feat = F.normalize(txt_feat)

            # reshape such that we can easily marginalize over irrelevant features
            embedding_dim = txt_feat.size(-1)
            txt_feat = txt_feat.reshape(*feature_shapes, embedding_dim)

        self.prompt_feat = txt_feat

    @torch.inference_mode()
    def _compute_img_feat(self, img: torch.Tensor) -> torch.Tensor:
        """Encode images using the given CLIP model."""
        assert img.ndim in [3, 4]
        img = img.unsqueeze(0) if img.ndim == 3 else img

        img_feat = self.clip.encode_image(img.to(self.device))
        assert img_feat.ndim == 2
        img_feat = F.normalize(img_feat)

        return img_feat

    @torch.inference_mode()
    def _compute_logits(self, img_feat: torch.Tensor) -> torch.Tensor:
        """Compute contraction between image and prompt features along embedding dimension."""
        # in shapes: (batch_size, embed_dim) x (embed_dim, *features)
        # out shape: (batch_size, *features)
        logits = torch.tensordot(img_feat, self.prompt_feat.movedim(-1, 0), dims=1)
        return logits

    @torch.inference_mode()
    def _compute_scores(self, img_feat: torch.Tensor) -> torch.Tensor:
        """Scale logits and compute softmax over feature dimensions."""
        logits = self.clip.logit_scale * self._compute_logits(img_feat)
        scores = F.softmax(logits.flatten(1), dim=1).reshape_as(logits)
        return scores

    @abstractmethod
    def variance_from_features(self, img_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute logit variance overall features and over individual features."""
        pass

    @abstractmethod
    def predict_from_features(self, img_feat: torch.Tensor, return_scores: bool = False) -> dict[str, torch.Tensor]:
        """Compute zero-shot predictions from given image features."""
        pass

    def predict(self, img: torch.Tensor, return_scores: bool = False) -> dict[str, torch.Tensor]:
        """Compute zero-shot predictions for given images."""
        return self.predict_from_features(self._compute_img_feat(img), return_scores=return_scores)


class ZeroShotClassifier(AbstractZeroShotClassifier):
    def __init__(
        self,
        clip: AbstractCLIP,
        tokenizer: SimpleTokenizer | HFTokenizer,
        idx2class: dict[int, str] | list[str],
        prompt_fn: Callable[[str], str] = identity,
    ) -> None:
        prompts = tokenizer([prompt_fn(idx2class[idx]) for idx in range(len(idx2class))])
        super().__init__(clip, prompts)

    def variance_from_features(self, img_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute logit variance overall features and over individual features."""
        scores = self._compute_logits(img_feat.to(self.device))

        variance = scores.var()

        return {'variance': variance}

    def predict_from_features(self, img_feat: torch.Tensor, return_scores: bool = False) -> dict[str, torch.Tensor]:
        """Compute zero-shot predictions from given image features."""
        scores = self._compute_logits(img_feat.to(self.device))

        pred = scores if return_scores else scores.argmax(dim=1)

        return {'pred': pred}


class OpenAIZeroShotClassifier(ZeroShotClassifier):
    templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
        'a clipart of the {}.',
        'a clipart of a {}.',
        'an infograph of the {}.',
        'an infograph of a {}.',
        'a quickdraw of the {}.',
        'a quickdraw of a {}.',
    ]

    def __init__(
        self,
        clip: AbstractCLIP,
        tokenizer: SimpleTokenizer | HFTokenizer,
        idx2class: dict[int, str] | list[str],
        domain_invariant: bool = False,
    ) -> None:
        # zero-shot model
        self.clip = clip
        self.clip.eval()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip.to(self.device)

        if domain_invariant:
            self.templates = [
                t
                for t in self.templates
                if any([d in t for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch']])
            ]

        classnames = [idx2class[idx] for idx in range(len(idx2class))]
        with torch.inference_mode():
            txt_feat = []
            for classname in classnames:
                texts = [template.format(classname) for template in self.templates]  # format with class
                texts = tokenizer(texts).to(self.device)  # tokenize

                class_embeddings = self.clip.encode_text(texts)  # embed with text encoder
                class_embeddings = F.normalize(class_embeddings, dim=-1)

                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = F.normalize(class_embedding, dim=-1)

                txt_feat.append(class_embedding)

            txt_feat = torch.stack(txt_feat, dim=0)

        self.prompt_feat = txt_feat
