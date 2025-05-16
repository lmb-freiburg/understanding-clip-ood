import gc
import math
import os
import random
from collections import defaultdict
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from nnsight import NNsight
from open_clip import get_tokenizer
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer
from torch import nn
from tqdm import tqdm

from xclip.datasets import DomainNetCaptions
from xclip.feature_circuits.circuit import compute_edges_new, compute_nodes
from xclip.feature_circuits.dictionary import IdentityDict
from xclip.feature_circuits.submodule import Submodule
from xclip.open_clip import OpenCLIP
from xclip.utils import AbstractCLIP
from xclip.zero_shot import OpenAIZeroShotClassifier

DEBUGGING = False
if DEBUGGING:
    tracer_kwargs = {'validate': True, 'scan': True}
else:
    tracer_kwargs = {'validate': False, 'scan': False}


class OpenAIZeroShotClassifierLocal(nn.Module):
    def __init__(
        self,
        model: AbstractCLIP,
        tokenizer: SimpleTokenizer | HFTokenizer,
        idx2class: dict[int, str] | list[str],
        # prompt_fn: Callable[[str], str] = identity,
    ) -> None:
        super().__init__()

        self.visual = model.clip.visual
        self.device = next(model.parameters()).device

        classnames = [idx2class[idx] for idx in range(len(idx2class))]
        with torch.no_grad():
            txt_feat = []
            for classname in classnames:
                texts = [
                    template.format(classname) for template in OpenAIZeroShotClassifier.templates
                ]  # format with class
                texts = tokenizer(texts).to(self.device)  # tokenize

                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings = F.normalize(class_embeddings, dim=-1)

                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = F.normalize(class_embedding, dim=-1)

                txt_feat.append(class_embedding)

            txt_feat = torch.stack(txt_feat, dim=0)

        self.normalized_txt_features = txt_feat

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Compute zero-shot predictions from given image features."""
        img_feat = self.visual(img.to(self.device))
        assert img_feat.ndim == 2
        normalized_img_feat = F.normalize(img_feat)
        logits = normalized_img_feat @ self.normalized_txt_features.T
        return logits


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_filepath = os.path.join(args.model_dir, 'checkpoints/epoch_32.pt')
    assert os.path.isfile(ckpt_filepath), f'Checkpoint file not found: {ckpt_filepath}'
    model, _, preprocess_val = OpenCLIP.from_pretrained(args.model, ckpt_path=ckpt_filepath)
    model.eval()
    model.to(args.device)
    tokenizer = get_tokenizer(args.model)

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
    all_data = {
        'real': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'infograph', 'painting', 'quickdraw', 'sketch'],
            # filter_classes={'real': list(ood_classes.values())},
        ),
        'quickdraw': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'infograph', 'painting', 'real', 'sketch'],
            # filter_classes={'quickdraw': list(ood_classes.values())},
        ),
        'sketch': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'infograph', 'painting', 'real', 'quickdraw'],
            # filter_classes={'sketch': list(ood_classes.values())},
        ),
        'clipart': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['infograph', 'painting', 'real', 'quickdraw', 'sketch'],
            # filter_classes={'clipart': list(ood_classes.values())},
        ),
        'infograph': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'painting', 'real', 'quickdraw', 'sketch'],
            # filter_classes={'infograph': list(ood_classes.values())},
        ),
        'painting': DomainNetCaptions(
            args.domainnet_path,
            'val',
            transform=preprocess_val,
            exclude_domains=['clipart', 'infograph', 'real', 'quickdraw', 'sketch'],
            # filter_classes={'painting': list(ood_classes.values())},
        ),
    }

    domainnet_classes = {}
    for path, label, _ in all_data['real'].samples:
        *_, clss, _ = path.split('/')
        domainnet_classes[label] = clss.replace('_', ' ')

    zero_shot_classifier = NNsight(OpenAIZeroShotClassifierLocal(model, tokenizer, domainnet_classes))

    # loading dictionaries
    stem = [
        Submodule(name='input', submodule=zero_shot_classifier.visual.conv1, use_input=True),
        Submodule(name='act1', submodule=zero_shot_classifier.visual.act1),
        Submodule(name='act2', submodule=zero_shot_classifier.visual.act2),
        Submodule(name='act3', submodule=zero_shot_classifier.visual.act3),
        Submodule(name='avgpool', submodule=zero_shot_classifier.visual.avgpool),
    ]
    offset = 1
    blocks = [
        Submodule(name=f'resblock{i + offset}', submodule=zero_shot_classifier.visual.layer1[i])
        for i in range(len(zero_shot_classifier.visual.layer1))
    ]
    offset += len(model.clip.visual.layer1)
    blocks += [
        Submodule(name=f'resblock{i + offset}', submodule=zero_shot_classifier.visual.layer2[i])
        for i in range(len(zero_shot_classifier.visual.layer2))
    ]
    offset += len(model.clip.visual.layer2)
    blocks += [
        Submodule(name=f'resblock{i + offset}', submodule=zero_shot_classifier.visual.layer3[i])
        for i in range(len(zero_shot_classifier.visual.layer3))
    ]
    offset += len(model.clip.visual.layer3)
    blocks += [
        Submodule(name=f'resblock{i + offset}', submodule=zero_shot_classifier.visual.layer4[i])
        for i in range(len(zero_shot_classifier.visual.layer4))
    ]
    attnpool = Submodule(name='attnpool', submodule=zero_shot_classifier.visual.attnpool)
    submodules_tmp = stem + blocks + [attnpool]

    # trick to be able to use the same code for the circuit analysis with only minor changes
    acts = []
    sample = next(iter(all_data[list(all_data.keys())[0]]))[0][None, ...].to(args.device).half()
    with torch.no_grad(), zero_shot_classifier.trace(sample, **tracer_kwargs):
        for submodule in submodules_tmp:
            acts.append(submodule.get_activation().save())
    all_submodules = [
        Submodule(
            name=submodule.name,
            submodule=submodule.submodule,
            use_input=submodule.use_input,
            is_tuple=submodule.is_tuple,
            shape=act.shape,
        )
        for submodule, act in zip(submodules_tmp, acts)
    ]
    del acts, submodules_tmp

    dictionaries = {}
    for submodule in all_submodules:
        if 'input' == submodule.name:
            num_features = submodule.submodule.in_channels
        elif 'act1' == submodule.name:
            num_features = zero_shot_classifier.visual.bn1.num_features
        elif 'act2' == submodule.name:
            num_features = zero_shot_classifier.visual.bn2.num_features
        elif 'act3' == submodule.name:
            num_features = zero_shot_classifier.visual.bn3.num_features
        elif 'avgpool' == submodule.name:
            num_features = zero_shot_classifier.visual.bn3.num_features
        elif 'resblock' in submodule.name:
            num_features = submodule.submodule.bn3.num_features
        elif 'attnpool' == submodule.name:
            num_features = zero_shot_classifier.visual.attnpool.c_proj.out_features
        else:
            raise NotImplementedError
        dictionaries[submodule] = IdentityDict(num_features)

    os.makedirs(os.path.join(args.model_dir, 'circuit_analysis'), exist_ok=True)

    pbar = tqdm(all_data.items(), leave=False, total=len(all_data), desc='Domain') if args.verbose else all_data.items()
    for domain, domain_data in pbar:
        if args.domain != 'all' and domain != args.domain:
            continue

        out_folder = os.path.join(args.model_dir, 'circuit_analysis', domain)
        os.makedirs(out_folder, exist_ok=True)

        label_to_indices = defaultdict(list)
        for idx, (_, label, _) in enumerate(domain_data.samples):
            label_to_indices[label].append(idx)
        label_to_indices = dict(label_to_indices)

        # pbar2 = tqdm(label_to_indices.items(), leave=False, desc='Class') if args.verbose else label_to_indices.items()
        ood_labels = list(ood_classes.values())
        id_labels = [label for label in label_to_indices.keys() if label not in ood_labels]
        all_labels = ood_labels + id_labels
        if args.class_idx is not None:
            all_labels = [args.class_idx]
        pbar2 = tqdm(all_labels, leave=False, desc='Class') if args.verbose else all_labels
        for label in pbar2:
            if os.path.exists(os.path.join(out_folder, f'{label}_edges.pt')) and not args.regenerate:
                continue
            indices = deepcopy(label_to_indices[label])
            random.shuffle(indices)
            images = []
            for idx in indices[: args.samples_per_class]:
                img, label = domain_data[idx]
                images.append(img)

            images = torch.stack(images, dim=0).half().to(args.device)
            num_examples = len(images)

            # Compute nodes
            batch_idx = 0
            batch_size = len(images)  # args.batch_size # compute all at once
            running_nodes, running_effects, running_deltas, running_grads, running_total_effect = (
                None,
                None,
                None,
                None,
                None,
            )
            while batch_idx < len(images):
                batch = images[batch_idx : min(batch_idx + batch_size, len(images))]
                labels = torch.tensor([label for _ in range(len(batch))]).to(args.device)

                def metric_fn(model, labels):
                    logits = model.output
                    return torch.gather(logits, dim=-1, index=labels.view(-1, 1)).squeeze(-1)

                metric_fn = partial(metric_fn, labels=labels)

                nodes, (effects, deltas, grads, total_effect) = compute_nodes(
                    clean=batch,
                    patch=None,
                    model=zero_shot_classifier,
                    all_submods=all_submodules,
                    dictionaries=dictionaries,
                    metric_fn=metric_fn,
                    aggregation='sum',  # or 'none' for not aggregating across sequence position
                    verbose=args.verbose,
                )

                if running_nodes is None:
                    running_nodes = {k: len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
                    running_effects = {k: len(batch) * effects[k].to('cpu') for k in effects.keys() if k.name != 'y'}
                    running_deltas = {k: len(batch) * deltas[k].to('cpu') for k in deltas.keys() if k.name != 'y'}
                    running_grads = {k: len(batch) * grads[k].to('cpu') for k in grads.keys() if k.name != 'y'}
                    if total_effect is not None:
                        running_total_effect = len(batch) * total_effect.to('cpu')
                else:
                    for submod in all_submodules:
                        if submod.name != 'y':
                            running_nodes[submod.name] += len(batch) * nodes[submod.name].to('cpu')
                            running_effects[submod.name] += len(batch) * effects[submod.name].to('cpu')
                            running_deltas[submod.name] += len(batch) * deltas[submod.name].to('cpu')
                            running_grads[submod.name] += len(batch) * grads[submod.name].to('cpu')
                    if total_effect is not None:
                        running_total_effect += len(batch) * total_effect.to('cpu')

                # memory cleanup
                del nodes, effects, deltas, grads, total_effect
                gc.collect()
                torch.cuda.empty_cache()

                batch_idx += len(batch)

            nodes = {k: v.to(args.device) / num_examples for k, v in running_nodes.items()}
            effects = {k: v.to(args.device) / num_examples for k, v in running_effects.items()}
            deltas = {k: v.to(args.device) / num_examples for k, v in running_deltas.items()}
            grads = {k: v.to(args.device) / num_examples for k, v in running_grads.items()}
            if running_total_effect is not None:
                total_effect = running_total_effect.to(args.device) / num_examples
            else:
                total_effect = None
            torch.save(nodes, os.path.join(out_folder, f'{label}_nodes.pt'))

            features_by_submod, features_by_submod_by_name = {}, {}
            for submod in all_submodules:
                num_neurons = effects[submod].act.size(-1)
                num_neurons_to_pick = math.ceil(num_neurons * args.node_threshold)
                features_by_submod[submod] = torch.topk(
                    effects[submod].sum(dim=1).mean(dim=0).abs().act, num_neurons_to_pick
                ).indices
                features_by_submod_by_name[submod.name] = torch.topk(
                    effects[submod].sum(dim=1).mean(dim=0).abs().act, num_neurons_to_pick
                ).indices
            torch.save(features_by_submod_by_name, os.path.join(out_folder, f'{label}_features_by_submod.pt'))

            # Compute edges
            running_edges = None
            batch_idx = 0
            while batch_idx < len(images):
                batch = images[batch_idx : min(batch_idx + batch_size, len(images))]
                labels = torch.tensor([label for _ in range(len(batch))]).to(args.device)

                def metric_fn(model, labels):
                    logits = model.output
                    return torch.gather(logits, dim=-1, index=labels.view(-1, 1)).squeeze(-1)

                metric_fn = partial(metric_fn, labels=labels)

                edges = compute_edges_new(
                    clean=batch,
                    patch=None,
                    model=zero_shot_classifier,
                    all_submods=all_submodules,
                    dictionaries=dictionaries,
                    features_by_submod=features_by_submod,
                    metric_fn=metric_fn,
                    aggregation='sum',  # or 'none' for not aggregating across sequence position
                    verbose=args.verbose,
                )

                for k in edges.keys():
                    for v in edges[k].keys():
                        edges[k][v] = edges[k][v].float()

                if running_edges is None:
                    running_edges = {
                        k: {kk: len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys()} for k in edges.keys()
                    }
                else:
                    for k in edges.keys():
                        for v in edges[k].keys():
                            running_edges[k][v] += len(batch) * edges[k][v].to('cpu')

                # memory cleanup
                del edges
                gc.collect()
                torch.cuda.empty_cache()

                batch_idx += len(batch)

            edges = {
                k: {kk: 1 / num_examples * v.to(args.device) for kk, v in running_edges[k].items()}
                for k in running_edges.keys()
            }
            torch.save(edges, os.path.join(out_folder, f'{label}_edges.pt'))

            del (
                nodes,
                edges,
                effects,
                deltas,
                grads,
                total_effect,
                running_nodes,
                running_edges,
                running_effects,
                running_deltas,
                running_grads,
                running_total_effect,
            )
            del features_by_submod, features_by_submod_by_name
            del images
            gc.collect()
            torch.cuda.empty_cache()

            # if label == 9:
            #     break

    print('Done.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Configure CLIP models for neuron analysis.')
    parser.add_argument('--model', type=str, required=True, help='CLIP model type')
    parser.add_argument('--model_dir', type=str, required=True, help='path to model directory')
    parser.add_argument(
        '--domain',
        type=str,
        default='all',
        choices=['real', 'quickdraw', 'sketch', 'clipart', 'infograph', 'painting', 'all'],
        help='domain to analyze',
    )
    parser.add_argument('--class_idx', type=int, default=None, help='class index to analyze')
    parser.add_argument('--domainnet_path', type=str, required=True, help='path to domainnet directory')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('--device', type=str, default='cuda', help='device to run on')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--samples_per_class', type=int, default=50, help='number of samples per class')
    parser.add_argument('--regenerate', action='store_true', help='regenerate attribution scores')

    parser.add_argument('--node_threshold', type=float, default=0.1, help='node threshold for pruning')
    parser.add_argument('--edge_threshold', type=float, default=0.01, help='edge threshold for pruning')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
