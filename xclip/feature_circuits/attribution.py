# Code from https://github.com/saprmarks/feature-circuits
from collections import namedtuple
from typing import Callable, Union

import torch as t
from nnsight.envoy import Envoy
from tqdm import tqdm

from xclip.feature_circuits.activation_utils import SparseAct
from xclip.feature_circuits.dictionary import Dictionary
from xclip.feature_circuits.submodule import Submodule

DEBUGGING = False
if DEBUGGING:
    tracer_kwargs = {'validate': True, 'scan': True}
else:
    tracer_kwargs = {'validate': False, 'scan': False}

EffectOut = namedtuple('EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])


def _pe_ig(
    clean,
    patch,
    model,
    submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
    verbose: bool = False,
):
    hidden_states_clean = {}
    with t.no_grad(), model.trace(clean, **tracer_kwargs):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save())  # type: ignore
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k: SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with t.no_grad(), model.trace(patch, **tracer_kwargs):
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.get_activation()
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseAct(act=f.save(), res=residual.save())  # type: ignore
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k: v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    pbar = tqdm(submodules, leave=False, desc='IG') if verbose else submodules
    for submodule in pbar:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.requires_grad_().retain_grad()
                f.res.requires_grad_().retain_grad()
                fs.append(f)
                with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                    submodule.set_activation(dictionary.decode(f.act) + f.res)
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True)  # type: ignore
            # torch.autograd.grad(metric.sum(), [f.act, f.res], retain_graph=True) # TODO: check if this is equivalent

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseAct(act=mean_grad, res=mean_residual_grad)  # type: ignore
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return EffectOut(effects, deltas, grads, total_effect)


def patching_effect(
    clean,
    patch,
    model,
    submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    metric_fn: Callable[[Envoy], t.Tensor],
    method='attrib',
    steps=10,
    metric_kwargs=dict(),
    verbose: bool = False,
):
    if method == 'ig':
        return _pe_ig(
            clean,
            patch,
            model,
            submodules,
            dictionaries,
            metric_fn,
            steps=steps,
            metric_kwargs=metric_kwargs,
            verbose=verbose,
        )
    else:
        raise ValueError(f'Unknown method {method}')


def jvp(
    input,
    model,
    dictionaries,
    downstream_submod,
    downstream_features,
    upstream_submod,
    left_vec: SparseAct,
    right_vec: SparseAct,
    intermediate_stopgrads: list[Submodule] = [],
    verbose: bool = False,
):
    downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]
    # b, s, n_feats = downstream_features.act.shape
    b, s, n_feats = right_vec.act.shape

    if t.all(downstream_features.to_tensor() == 0):
        return t.sparse_coo_tensor(
            t.zeros((2 * downstream_features.act.dim(), 0), dtype=t.long),
            t.zeros(0),
            size=(b, s, n_feats + 1, b, s, n_feats + 1),
        ).to(model.device)

    vjv_values = {}

    pbar = (
        tqdm(downstream_features.to_tensor().nonzero(), leave=False)
        if verbose
        else downstream_features.to_tensor().nonzero()
    )
    for downstream_feat_idx in pbar:
        # print(downstream_feat_idx)
        with model.trace(input, **tracer_kwargs):
            # forward pass modifications
            x = upstream_submod.get_activation()
            x_hat, f = upstream_dict.hacked_forward_for_sfc(x)  # hacking around an nnsight bug
            x_res = x - x_hat
            upstream_submod.set_activation(x_hat + x_res)
            upstream_act = SparseAct(act=f, res=x_res).save()

            y = downstream_submod.get_activation()
            y_hat, g = downstream_dict.hacked_forward_for_sfc(y)  # hacking around an nnsight bug
            y_res = y - y_hat
            downstream_submod.set_activation(y_hat + y_res)
            downstream_act = SparseAct(act=g, res=y_res).save()

            to_backprops = (left_vec @ downstream_act).to_tensor()

            # stop grad
            for submodule in intermediate_stopgrads:
                submodule.stop_grad()
            x_res.grad = t.zeros_like(x_res.grad)

            vjv = (upstream_act.grad @ right_vec).to_tensor()
            # to_backprops[tuple(downstream_feat_idx)].backward(retain_graph=True)
            to_backprops[tuple(downstream_feat_idx)].backward(retain_graph=False)  # TODO: double check

            vjv_values[downstream_feat_idx] = vjv.save()

    vjv_indices = t.stack(list(vjv_values.keys()), dim=0).T
    vjv_values = t.stack([v.value for v in vjv_values.values()], dim=0)

    return t.sparse_coo_tensor(vjv_indices, vjv_values, size=(b, s, n_feats + 1, b, s, n_feats + 1))


# aggregate over sequence positions/pixels


def jvp_new(
    input,
    model,
    dictionaries,
    downstream_submod,
    downstream_features,
    upstream_submod,
    left_vec: SparseAct,
    right_vec: SparseAct,
    intermediate_stopgrads: list[Submodule] = [],
    verbose: bool = False,
):
    downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]
    # b, s, n_feats = downstream_features.act.shape
    b, s, n_feats = right_vec.act.shape

    # if t.all(downstream_features.to_tensor() == 0):
    if len(downstream_features) == 0:
        raise NotImplementedError
        return t.sparse_coo_tensor(
            t.zeros((2 * downstream_features.act.dim(), 0), dtype=t.long),
            t.zeros(0),
            size=(b, n_feats + 1, b, n_feats + 1),
        ).to(model.device)

    vjv_values = {}

    # pbar = tqdm(downstream_features.to_tensor().nonzero()[:1], leave=False) if verbose else downstream_features.to_tensor().nonzero()
    pbar = tqdm(downstream_features, leave=False) if verbose else downstream_features
    for downstream_feat_idx in pbar:
        # print(downstream_feat_idx)
        with model.trace(input, **tracer_kwargs):
            # forward pass modifications
            x = upstream_submod.get_activation()
            x_hat, f = upstream_dict.hacked_forward_for_sfc(x)  # hacking around an nnsight bug
            x_res = x - x_hat
            upstream_submod.set_activation(x_hat + x_res)
            upstream_act = SparseAct(act=f, res=x_res).save()

            y = downstream_submod.get_activation()
            y_hat, g = downstream_dict.hacked_forward_for_sfc(y)  # hacking around an nnsight bug
            y_res = y - y_hat
            downstream_submod.set_activation(y_hat + y_res)
            downstream_act = SparseAct(act=g, res=y_res).save()

            to_backprops = (left_vec @ downstream_act).to_tensor()

            # stop grad
            for submodule in intermediate_stopgrads:
                submodule.stop_grad()
            x_res.grad = t.zeros_like(x_res.grad)

            vjv = (upstream_act.grad @ right_vec).to_tensor()
            # to_backprops[tuple(downstream_feat_idx)].backward(retain_graph=True)
            # to_backprops[0, :, 0].mean().backward(retain_graph=False) # aggregate over sequence positions
            to_backprops[..., downstream_feat_idx].mean().backward(
                retain_graph=False
            )  # aggregate over batch dim and seq dim

            # vjv_values[downstream_feat_idx] = vjv.cpu().save()
            vjv_values[downstream_feat_idx.cpu()] = vjv.cpu().save()
            # vjv_values[tuple(downstream_feat_idx.cpu().numpy())] = vjv.cpu().save()
            # a = vjv.save()

    # vjv_indices = t.stack(list(vjv_values.keys()), dim=0).T
    # vjv_indices = t.stack(list(vjv_values.keys()), dim=0)
    vjv_values = t.stack(
        [v.value for v in vjv_values.values()], dim=0
    )  # len(downstream_features) x b x s x (n_feats+1)

    # return t.sparse_coo_tensor(vjv_indices, vjv_values, size=(b, s, n_feats+1, b, s, n_feats+1))
    # return t.sparse_coo_tensor(vjv_indices, vjv_values, size=(len(downstream_features), b, s, n_feats+1))
    return vjv_values


def upstream_neuron_attribution(
    clean,
    patch,
    model,
    upstream_submodule: Submodule,
    downstream_submodule: Submodule,
    # submodules: list[Submodule],
    dictionaries: dict[Submodule, Dictionary],
    upstream_neurons: Union[list, t.Tensor],
    downstream_neurons: Union[list, t.Tensor],
    steps=10,
    verbose: bool = False,
    method: str = 'ig',
):
    if method == 'ig':
        hidden_states_clean = {}
        with t.no_grad(), model.trace(clean, **tracer_kwargs):
            for submodule in [upstream_submodule, downstream_submodule]:
                dictionary = dictionaries[submodule]
                x = submodule.get_activation()
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_clean[submodule] = SparseAct(act=f.save(), res=residual.save())  # type: ignore
            # metric_clean = metric_fn(model, **metric_kwargs).save()
        hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

        if patch is None:
            hidden_states_patch = {
                k: SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)) for k, v in hidden_states_clean.items()
            }
            # total_effect = None
        else:
            raise NotImplementedError
            # hidden_states_patch = {}
            # with t.no_grad(), model.trace(patch, **tracer_kwargs):
            #     for submodule in [upstream_submodule, downstream_submodule]:
            #         dictionary = dictionaries[submodule]
            #         x = submodule.get_activation()
            #         f = dictionary.encode(x)
            #         x_hat = dictionary.decode(f)
            #         residual = x - x_hat
            #         hidden_states_patch[submodule] = SparseAct(act=f.save(), res=residual.save())  # type: ignore
            #     metric_patch = metric_fn(model, **metric_kwargs).save()
            # total_effect = (metric_patch.value - metric_clean.value).detach()
            # hidden_states_patch = {k: v.value for k, v in hidden_states_patch.items()}

        # compute edge effects
        downstream_dictionary = dictionaries[downstream_submodule]
        downstream_clean_state = hidden_states_clean[downstream_submodule]
        upstream_dictionary = dictionaries[upstream_submodule]
        upstream_clean_state = hidden_states_clean[upstream_submodule]
        upstream_patch_state = hidden_states_patch[upstream_submodule]
        edge_effects = t.zeros(len(downstream_neurons), len(upstream_neurons))
        for d_idx, downstream_neuron in tqdm(
            enumerate(downstream_neurons), leave=False, desc='Downstream Neurons', total=len(downstream_neurons)
        ):
            with model.trace(**tracer_kwargs) as tracer:
                metrics = []
                fs = []
                for step in range(steps):
                    alpha = step / steps
                    f = (1 - alpha) * upstream_clean_state + alpha * upstream_patch_state
                    f.act.requires_grad_().retain_grad()
                    f.res.requires_grad_().retain_grad()
                    fs.append(f)
                    with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                        upstream_submodule.set_activation(upstream_dictionary.decode(f.act) + f.res)
                        # metrics.append(metric_fn(model, **metric_kwargs))
                        x = downstream_submodule.get_activation()
                        down_f = downstream_dictionary.encode(x)
                        # score = (downstream_clean_state[..., downstream_neuron] - f[..., downstream_neuron])**2
                        score = t.nn.functional.mse_loss(
                            downstream_clean_state[..., downstream_neuron], down_f[..., downstream_neuron]
                        )
                        metrics.append(score)
                metric = sum([m for m in metrics])
                metric.sum().backward(retain_graph=True)  # type: ignore

            mean_grad = sum([f.act.grad for f in fs]) / steps
            mean_residual_grad = sum([f.res.grad for f in fs]) / steps
            grad = SparseAct(act=mean_grad, res=mean_residual_grad)  # type: ignore
            delta = (
                (upstream_patch_state - upstream_clean_state).detach()
                if upstream_patch_state is not None
                else -upstream_clean_state.detach()
            )
            effect = grad @ delta
            for u_idx, upstream_neuron in enumerate(upstream_neurons):
                edge_effects[d_idx, u_idx] = effect.act.sum(dim=1).mean(dim=0)[upstream_neuron].item()

            del f, fs, metric, grad, delta, effect
            t.cuda.empty_cache()

    elif method == 'attrib':
        raise NotImplementedError

    return edge_effects
