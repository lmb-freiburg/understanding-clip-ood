# Code from https://github.com/saprmarks/feature-circuits
import gc
from collections import defaultdict

import torch
from tqdm import tqdm

from xclip.feature_circuits.attribution import jvp_new, patching_effect, upstream_neuron_attribution
from xclip.feature_circuits.coo_utils import sparse_reshape

DEBUGGING = False
if DEBUGGING:
    tracer_kwargs = {'validate': True, 'scan': True}
else:
    tracer_kwargs = {'validate': False, 'scan': False}


def compute_nodes(
    clean,
    patch,
    model,
    all_submods,
    # embed,
    # attns,
    # mlps,
    # resids,
    dictionaries,
    metric_fn,
    metric_kwargs=dict(),
    aggregation='sum',  # or 'none' for not aggregating across sequence position
    verbose=False,
):
    # all_submods = (
    #     [embed] if embed is not None else []
    # ) + [
    #     submod for layer_submods in zip(attns, mlps, resids) for submod in layer_submods
    # ]

    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method='ig',  # get better approximations for early layers by using ig
        verbose=verbose,
    )

    # n_layers = len(resids)
    # nodes = {'y' : total_effect}
    # if embed is not None:
    #     nodes['embed'] = effects[embed]
    # for i in range(n_layers):
    #     nodes[f'attn_{i}'] = effects[attns[i]]
    #     nodes[f'mlp_{i}'] = effects[mlps[i]]
    #     nodes[f'resid_{i}'] = effects[resids[i]]
    # n_layers = len(all_submods)
    nodes = {'y': total_effect}
    for submod in all_submods:
        nodes[submod.name] = effects[submod]

    if aggregation == 'sum':
        for k in nodes:
            if k != 'y':
                nodes[k] = nodes[k].sum(dim=1)
    nodes = {k: v.mean(dim=0) for k, v in nodes.items() if k != 'y'}
    return nodes, (effects, deltas, grads, total_effect)


def compute_edges(
    clean,
    patch,
    model,
    all_submods,
    # embed,
    # attns,
    # mlps,
    # resids,
    dictionaries,
    features_by_submod,
    effects,
    deltas,
    grads,
    metric_fn,
    metric_kwargs=dict(),
    aggregation='sum',  # or 'none' for not aggregating across sequence position
    nodes_only=False,
    parallel_attn=False,
    node_threshold=0.1,
    verbose=False,
):
    edges = defaultdict(lambda: {})
    # edges[f'resid_{len(resids)-1}'] = { 'y' : effects[resids[-1]].to_tensor().flatten().to_sparse() }
    edges[all_submods[-1].name] = {'y': effects[all_submods[-1]].to_tensor().flatten().to_sparse()}

    def N(upstream, downstream, midstream=[]):
        result = jvp_new(
            clean,
            model,
            dictionaries,
            downstream,
            features_by_submod[downstream],
            upstream,
            grads[downstream],
            deltas[upstream],
            intermediate_stopgrads=midstream,
            verbose=verbose,
        )
        return result

    # now we work backward through the model to get the edges
    # pbar = tqdm(reversed(range(len(resids))), leave=False, desc="Inter-node effects") if verbose else reversed(range(len(resids)))
    pbar = (
        tqdm(reversed(range(2, len(all_submods))), total=len(all_submods) - 2, leave=False, desc='Inter-node effects')
        if verbose
        else reversed(range(len(all_submods)))
    )
    for layer in pbar:
        # resid = resids[layer]
        # mlp = mlps[layer]
        # attn = attns[layer]

        # MR_effect = N(mlp, resid)
        # AR_effect = N(attn, resid, [mlp])
        # edges[f'mlp_{layer}'][f'resid_{layer}'] = MR_effect
        # edges[f'attn_{layer}'][f'resid_{layer}'] = AR_effect

        # if parallel_attn:
        #     AM_effect = N(attn, mlp)
        #     edges[f'attn_{layer}'][f'mlp_{layer}'] = AM_effect

        # if layer > 0:
        #     prev_resid = resids[layer-1]
        # else:
        #     prev_resid = embed

        # if prev_resid is not None:
        #     RM_effect = N(prev_resid, mlp, [attn])
        #     RA_effect = N(prev_resid, attn)
        #     RR_effect = N(prev_resid, resid, [mlp, attn])

        #     if layer > 0:
        #         edges[f'resid_{layer-1}'][f'mlp_{layer}'] = RM_effect
        #         edges[f'resid_{layer-1}'][f'attn_{layer}'] = RA_effect
        #         edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect
        #     else:
        #         edges['embed'][f'mlp_{layer}'] = RM_effect
        #         edges['embed'][f'attn_{layer}'] = RA_effect
        #         edges['embed'][f'resid_0'] = RR_effect

        cur_layer = all_submods[layer]
        prev_layer = all_submods[layer - 1]
        if prev_layer is not None:
            RR_effect = N(prev_layer, cur_layer)
            indices = features_by_submod[prev_layer].cpu()
            edges[prev_layer.name][cur_layer.name] = RR_effect[..., indices]

    # rearrange weight matrices
    for child in edges:
        # get shape for child
        # bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == 'y':
                ch = [s for s in all_submods if s.name == child][0]
                bc, sc, fc = effects[ch].act.shape
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc + 1))
            else:
                continue
                # bp, sp, fp = nodes[parent].act.shape
                # assert bp == bc
                # try:
                #     weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
                # except:
                #     breakpoint()
            edges[child][parent] = weight_matrix

    if aggregation == 'sum':
        # aggregate across sequence position or spatial dims
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    # weight_matrix = weight_matrix.sum(dim=(1, 4))
                    weight_matrix = weight_matrix.sum(dim=(2))
                edges[child][parent] = weight_matrix
        # for node in nodes:
        #     if node != 'y':
        #         nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            # bc, fc = nodes[child].act.shape
            ch = [s for s in all_submods if s.name == child][0]
            bc, _, fc = effects[ch].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    pa = [s for s in all_submods if s.name == parent][0]
                    bp, _, fp = effects[pa].act.shape
                    assert bp == bc
                    # weight_matrix = weight_matrix.sum(dim=(0,2)) / bc
                    weight_matrix = weight_matrix.sum(dim=(1)) / bc
                edges[child][parent] = weight_matrix
        # for node in nodes:
        #     if node != 'y':
        #         nodes[node] = nodes[node].mean(dim=0)

    elif aggregation == 'none':
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown aggregation: {aggregation}')

    return edges


def compute_edges_new(
    clean,
    patch,
    model,
    all_submods,
    # embed,
    # attns,
    # mlps,
    # resids,
    dictionaries,
    features_by_submod,
    # effects,
    # deltas,
    # grads,
    metric_fn,
    metric_kwargs=dict(),
    aggregation='sum',  # or 'none' for not aggregating across sequence position
    edge_threshold=0.1,
    verbose=False,
):
    edges = defaultdict(lambda: {})
    # edges[f'resid_{len(resids)-1}'] = { 'y' : effects[resids[-1]].to_tensor().flatten().to_sparse() }
    # edges[all_submods[-1].name] = { 'y' : effects[all_submods[-1]].to_tensor().flatten().to_sparse() }

    # now we work backward through the model to get the edges
    # pbar = tqdm(reversed(range(len(resids))), leave=False, desc="Inter-node effects") if verbose else reversed(range(len(resids)))
    pbar = (
        tqdm(reversed(range(2, len(all_submods))), total=len(all_submods) - 2, leave=False, desc='Inter-node effects')
        if verbose
        else reversed(range(len(all_submods)))
    )
    for layer in pbar:
        cur_layer = all_submods[layer]
        prev_layer = all_submods[layer - 1]
        if prev_layer is not None:
            cur_layer_name = cur_layer.name
            prev_layer_name = prev_layer.name
            cur_edges = upstream_neuron_attribution(
                clean=clean,
                patch=patch,
                model=model,
                upstream_submodule=prev_layer,
                downstream_submodule=cur_layer,
                dictionaries=dictionaries,
                upstream_neurons=features_by_submod[prev_layer],
                downstream_neurons=features_by_submod[cur_layer],
                method='ig',
            )
            edges[prev_layer_name][cur_layer_name] = cur_edges.cpu().clone()

            del cur_edges
            gc.collect()
            torch.cuda.empty_cache()

    return edges
