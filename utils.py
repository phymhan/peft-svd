import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from functools import partial

SAFETENSORS_WEIGHTS_EXTENSION = ".safetensors"
WEIGHTS_EXTENSION = ".bin"

def is_weight_module(m, n):
    return hasattr(m, 'weight') and isinstance(getattr(m, 'weight', None), torch.Tensor)


class _SpectralShift(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        dim: int = 0,
        cached_svd_params: dict = None,
        svd_rank: int = None,  # if None, do not limit rank
        svd_shift_init_scale: float = 0.,
        svd_no_residual: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        self.shape = weight.shape
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")
        self.dim = dim if dim >= 0 else dim + ndim
        self.svd_rank = None if (svd_rank is not None and svd_rank < 0) else svd_rank
        self.use_original = False
        self.eigenvalues_delta_cached = None
        if ndim > 1:
            if cached_svd_params is None:
                weight_mat = self._reshape_weight_to_matrix(weight)
                with torch.no_grad():
                    u, s, vh = torch.linalg.svd(weight_mat, full_matrices=False)
                residual = weight_mat - u @ torch.diag(s) @ vh
                if svd_no_residual:
                    residual.zero_()
            else:
                u = cached_svd_params['u']
                s = cached_svd_params['s']
                vh = cached_svd_params['vh']
                residual = cached_svd_params['residual']
            self.register_buffer('_u', u.detach())
            self.register_buffer('_s', s.detach())
            self.register_buffer('_vh', vh.detach())
            self.register_buffer('_residual', residual.detach())
            eigenvalues_delta = torch.randn_like(s) * svd_shift_init_scale
            if self.svd_rank is not None:
                if eigenvalues_delta.ndim > 1:
                    eigenvalues_delta = eigenvalues_delta[:,:self.svd_rank]
                else:
                    eigenvalues_delta = eigenvalues_delta[:self.svd_rank]
                svd_shift_index = torch.arange(len(s))
                self.svd_shift_index = svd_shift_index[:self.svd_rank]
        else:
            eigenvalues_delta = torch.randn_like(weight) * svd_shift_init_scale
        self.eigenvalues_delta = nn.Parameter(eigenvalues_delta, requires_grad=True)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))

        return weight.flatten(1)  # NOTE: for conv2d it will be (c_out, c_in * k_x * k_y)
    
    def _reshape_matrix_to_weight(self, weight_mat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        if self.dim != 0:
            # dim permuted to front
            weight = weight_mat.reshape(shape[self.dim], *(shape[d] for d in range(len(shape)) if d != self.dim))
            weight = weight.permute(*np.argsort([self.dim] + [d for d in range(weight.dim()) if d != self.dim]))
        else:
            weight = weight_mat.reshape(shape)

        return weight

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.use_original:
            return weight
        eigenvalues_delta = self.eigenvalues_delta if self.eigenvalues_delta_cached is None else self.eigenvalues_delta_cached
        if weight.ndim == 1:
            delta = eigenvalues_delta
            return weight.detach() + delta
        else:
            if self.svd_rank is not None:
                delta = torch.zeros_like(self._s)
                delta[self.svd_shift_index] = eigenvalues_delta
            else:
                delta = eigenvalues_delta
            weight_mat = self._u.detach() @ torch.diag(F.relu(self._s.detach() + delta)) @ self._vh.detach()
            weight_mat = weight_mat + self._residual.detach()
            return self._reshape_matrix_to_weight(weight_mat, weight.shape)
    
    def __repr__(self):
        return f"(Spectral Shift) shape={self.shape}, dim={self.dim}, rank={self.svd_rank}"


def spectral_shift(module, name='weight', dim=None, cached_svd_params=None, svd_kwargs=None):
    r"""
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter. Default: ``"weight"``.
        dim (int, optional): dimension corresponding to number of outputs.
            Default: ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with a new parametrization registered to the specified
        weight
    """
    if svd_kwargs is None:
        svd_kwargs = {}
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    new_module = _SpectralShift(weight, dim=dim,
        cached_svd_params=cached_svd_params, **svd_kwargs)
    nn.utils.parametrize.register_parametrization(module, name, new_module)
    return module, new_module  # NOTE: module is parametrized inplace


def check_weight_ndim(weight, ndims=None):
    if ndims is None:
        return True
    elif weight.ndim in ndims:
        return True
    else:
        return False


def register_spectral_shift(
    model,
    cached_svd_state_dict=None,
    bias_trainable=True, svd_kwargs=None,
    parametrized_module_list=[],
    new_module_dict={'1d': [], '2d': [], '3d': [], '4d': [], 'bias': []},
    new_module_params={'1d': [], '2d': [], '3d': [], '4d': [], 'bias': []},
    layer_selection=None,
):
    # ==========
    model.cuda()  # NOTE: hardcoded, always move to cuda for SVD computation
    # ==========

    weight_ndim_checker = partial(check_weight_ndim, ndims=[1, 2, 4])

    device = next(model.parameters()).device
    if cached_svd_state_dict is not None and 'state_dict' in cached_svd_state_dict:
        cached_svd_state_dict = cached_svd_state_dict['state_dict']
    for n, m in model.named_modules():
        n_ = n+'.weight'
        if is_weight_module(m, n_):
            # print(n, m.weight.shape)
            if not weight_ndim_checker(m.weight):
                continue
            if cached_svd_state_dict is not None and n + '.parametrizations.weight.0._s' in cached_svd_state_dict:
                cached_svd_params = {
                    'u': cached_svd_state_dict[n + '.parametrizations.weight.0._u'].to(device),
                    's': cached_svd_state_dict[n + '.parametrizations.weight.0._s'].to(device),
                    'vh': cached_svd_state_dict[n + '.parametrizations.weight.0._vh'].to(device),
                    'residual': cached_svd_state_dict[n + '.parametrizations.weight.0._residual'].to(device),
                }
            else:
                cached_svd_params = None
            m, new_m = spectral_shift(m, 'weight', dim=0, 
                cached_svd_params=cached_svd_params, svd_kwargs=svd_kwargs)
            parametrized_module_list.append(m)
            new_module_dict[f'{m.weight.ndim}d'].append(new_m)
            new_module_params[f'{m.weight.ndim}d'] += list(new_m.parameters())
            if bias_trainable and hasattr(m, 'bias') and m.bias is not None:
                new_module_params['bias'] += [m.bias]
    return parametrized_module_list, new_module_dict, new_module_params


def get_ndim_wise_params(
    model,
    module_params={'1d': [], '2d': [], '3d': [], '4d': [], 'bias': []},
    separate_bias=False,
):
    for n, m in model.named_modules():
        n_ = n + '.weight'
        if is_weight_module(m, n_):
            module_params[f'{m.weight.ndim}d'] += [m.weight]
            if hasattr(m, 'bias') and m.bias is not None:
                if separate_bias:
                    module_params['bias'] += [m.bias]
                else:
                    module_params[f'{m.weight.ndim}d'] += [m.bias]
            # module_params[f'{m.weight.ndim}d'] += list(m.parameters())  # NOTE: bias is included
    return module_params


"""
Utility functions for saving and loading SVD delta checkpoints
"""
def get_svd_delta_state_dict(model):
    bn_state_dict = {}
    state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if 'eigenvalues_delta' in k:
            new_state_dict[k] = state_dict[k]
        if 'running' in k:
            bn_state_dict[k] = state_dict[k]
    return new_state_dict, bn_state_dict


def get_weight_matrix_shape(weight, dim=0):
    shape = weight.shape
    if dim != 0:
        shape = [shape[dim]] + [shape[d] for d in range(len(shape)) if d != dim]
    return [shape[0], np.prod(shape[1:])]


def reshape_weight_to_matrix(weight: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert weight.ndim > 1
        if dim != 0:
            # permute dim to front
            weight = weight.permute(dim, *(d for d in range(weight.dim()) if d != dim))
        return weight.flatten(1)  # NOTE: for conv2d it will be (c_out, c_in * k_x * k_y)


def reshape_matrix_to_weight(weight_mat: torch.Tensor, shape: torch.Size, dim: int = 0) -> torch.Tensor:
    if dim != 0:
        # dim permuted to front
        weight = weight_mat.reshape(shape[dim], *(shape[d] for d in range(len(shape)) if d != dim))
        weight = weight.permute(*np.argsort([dim] + [d for d in range(weight.dim()) if d != dim]))
    else:
        weight = weight_mat.reshape(shape)
    return weight


def scale_svd_shifts(
    ckpt_svd,
    ckpt_delta,
    svd_delta_scale_1d=1.,
    svd_delta_scale_2d=1.,
    svd_delta_scale_3d=1.,
    svd_delta_scale_4d=1.,
):
    state_dict = {}
    names_delta = [k for k in ckpt_delta['state_dict'].keys() if 'parametrizations' in k]
    names_delta = list(set([n.split('parametrizations')[0] for n in names_delta]))
    names_bias = [k for k in ckpt_delta['state_dict'].keys() if k.endswith('.bias')]
    for k in names_delta:
        scale = 1.
        name_original = k + 'parametrizations.weight.original'
        name_delta = k + 'parametrizations.weight.0.eigenvalues_delta'
        weight = ckpt_svd['state_dict'][name_original]
        if weight.ndim == 1:
            scale *= svd_delta_scale_1d
        elif weight.ndim == 2:
            scale *= svd_delta_scale_2d
        elif weight.ndim == 3:
            scale *= svd_delta_scale_3d
        elif weight.ndim == 4:
            scale *= svd_delta_scale_4d
        # else:
        #     scale *= svd_delta_scale
        state_dict[name_delta] = ckpt_delta['state_dict'][name_delta] * scale
    for k in names_bias:
        bias_delta = ckpt_delta['state_dict'][k] - ckpt_svd['state_dict'][k]
        state_dict[k] = ckpt_svd['state_dict'][k] + svd_delta_scale_1d * bias_delta
    return {'state_dict': state_dict}


def scale_weight_deltas(
    ckpt_orig,
    ckpt,
    weight_delta_scale=1.,
):
    if weight_delta_scale == 1:
        return ckpt
    if 'state_dict' not in ckpt:
        ckpt = {'state_dict': ckpt}
    state_dict = {}
    for k in ckpt['state_dict'].keys():
        delta = ckpt['state_dict'][k] - ckpt_orig['state_dict'][k]
        state_dict[k] = ckpt_orig['state_dict'][k] + weight_delta_scale * delta
    return {'state_dict': state_dict}


def convert_svd_shift_to_weight(
    ckpt_svd,
    ckpt_delta,
    svd_delta_scale=1.,
    svd_delta_scale_1d=1.,
    svd_delta_scale_2d=1.,
    svd_delta_scale_3d=1.,
    svd_delta_scale_4d=1.,
    svd_kwargs=None,
):
    """ Combine deltas back to full weights
    """
    if svd_kwargs is None:
        svd_kwargs = {}
    names_all = [k for k in ckpt_svd['state_dict'].keys() if 'parametrizations' in k]
    names_all = list(set([n.split('parametrizations')[0] for n in names_all]))
    names_delta = [k for k in ckpt_delta['state_dict'].keys() if 'parametrizations' in k]
    names_delta = list(set([n.split('parametrizations')[0] for n in names_delta]))
    names_bias = [k for k in ckpt_delta['state_dict'].keys() if k.endswith('.bias')]
    svd_delta_scale_lut = {
        '1d': svd_delta_scale_1d,
        '2d': svd_delta_scale_2d,
        '3d': svd_delta_scale_3d,
        '4d': svd_delta_scale_4d,
    }
    for k in tqdm(names_delta):
        name = k + 'parametrizations.weight.original'
        weight = ckpt_svd['state_dict'][name]
        device = weight.device
        if weight.ndim > 1:
            u = ckpt_svd['state_dict'][k + 'parametrizations.weight.0._u'].cuda()
            s = ckpt_svd['state_dict'][k + 'parametrizations.weight.0._s'].cuda()
            vh = ckpt_svd['state_dict'][k + 'parametrizations.weight.0._vh'].cuda()
            if svd_kwargs.get('svd_no_residual', False):
                residual = 0
            else:
                residual = ckpt_svd['state_dict'][k + 'parametrizations.weight.0._residual'].cuda()
            delta = ckpt_delta['state_dict'][k + 'parametrizations.weight.0.eigenvalues_delta'].cuda()
            if delta.ndim > 1:
                if svd_kwargs.get('svd_normalize', False):
                    delta = F.normalize(delta, dim=1)
                delta = delta.sum(dim=0)
            else:
                if svd_kwargs.get('svd_normalize', False):
                    delta = F.normalize(delta, dim=0)
            if (svd_kwargs.get('svd_rank', None) is not None and
                svd_kwargs.get('svd_rank_space', '2d') in ["all", "2d"]
            ):
                delta[svd_kwargs['svd_rank']:] = 0.
            delta = delta * svd_delta_scale_lut.get(f'{weight.ndim}d', svd_delta_scale)
            if delta.shape != s.shape:
                tmp = torch.zeros_like(s)
                tmp[:delta.shape[0]] = delta
                delta = tmp

            with torch.no_grad():
                weight_mat = u @ torch.diag(F.relu(s + delta)) @ vh + residual
                weight = reshape_matrix_to_weight(weight_mat, weight.shape, 0)  # NOTE: dim=0 is hardcoded
            ckpt_svd['state_dict'][k + 'weight'] = weight.to(device)
            del ckpt_svd['state_dict'][k + 'parametrizations.weight.0._u']
            del ckpt_svd['state_dict'][k + 'parametrizations.weight.0._s']
            del ckpt_svd['state_dict'][k + 'parametrizations.weight.0._vh']
            del ckpt_svd['state_dict'][k + 'parametrizations.weight.0._residual']
            del ckpt_svd['state_dict'][k + 'parametrizations.weight.original']
        else:
            if svd_kwargs.get('svd_legacy', False):
                name = k + 'parametrizations.weight.0.eigenvalues_delta'
                ckpt_svd['state_dict'][k + 'weight'] = ckpt_delta['state_dict'][name]
            else:
                name = k + 'parametrizations.weight.0.eigenvalues_delta'
                delta = ckpt_delta['state_dict'][name]
                if delta.ndim > 1:
                    delta = delta.sum(dim=0)
                if (svd_kwargs.get('svd_rank', None) is not None and
                    svd_kwargs.get('svd_rank_space', '2d') in ["all", "1d"]
                ):
                    delta[svd_kwargs['svd_rank']:] = 0.
                ckpt_svd['state_dict'][k + 'weight'] = delta * svd_delta_scale_1d + \
                    ckpt_svd['state_dict'][k + 'parametrizations.weight.original']
    for k in names_bias:
        ckpt_svd['state_dict'][k] = ckpt_delta['state_dict'][k]
    for k in list(set(names_all) - set(names_delta) - set(names_bias)):
        # NOTE: recover non-parametrized weights to original
        name = k + 'parametrizations.weight.original'
        ckpt_svd['state_dict'][k + 'weight'] = ckpt_svd['state_dict'][name]
        del ckpt_svd['state_dict'][name]
    return ckpt_svd


def str2float(ss):
    if isinstance(ss, str):
        return [float(s) for s in re.split(",|:|;", ss)]
    elif isinstance(ss, (list, tuple)):
        return list(ss)
    else:
        raise ValueError


def load_svd_ckpt(
    model,
    ckpt_paths,
    ckpt_svd_path=None,
    ckpt_orig_path=None,
    weight_kwargs={},
    how_to_load="sequential",
    verbose=False,
):
    """
    If how_to_load == 'sequential', state dicts in ckpt_paths will be sequentially
    updated; If how_to_load == 'merge', weight deltas will be merged.
    """
    load_svd_delta = False
    if isinstance(ckpt_paths, str):
        ckpt_paths = re.split(",|:|;", ckpt_paths)
    # NOTE: try to automatically determine if ckpt is svd delta
    ckpt_size_in_bytes = os.path.getsize(ckpt_paths[-1])
    if ckpt_size_in_bytes <= 4e7:  # NOTE: hardcoded
        load_svd_delta = True
        print(f"size of last ckpt is {ckpt_size_in_bytes} bytes, load as delta...")

    if load_svd_delta:  # NOTE: load ckpts as SVD deltas
        svd_delta_scales_1d = str2float(weight_kwargs.get('svd_delta_scales_1d', [1.]))
        svd_delta_scales_2d = str2float(weight_kwargs.get('svd_delta_scales_2d', [1.]))
        svd_delta_scales_3d = str2float(weight_kwargs.get('svd_delta_scales_3d', [1.]))
        svd_delta_scales_4d = str2float(weight_kwargs.get('svd_delta_scales_4d', [1.]))
        # NOTE: scale deltas by scales
        print(f"Loading model from {ckpt_svd_path}")
        ckpt_svd = torch.load(ckpt_svd_path, map_location="cpu")
        if 'state_dict' not in ckpt_svd:
            ckpt_svd = {'state_dict': ckpt_svd}
        print(f"Loading deltas from {ckpt_paths}")
        ckpts_delta = [torch.load(ckpt_path, map_location="cpu") for ckpt_path in ckpt_paths]
        ckpts_delta = [ckpt if 'state_dict' in ckpt else {'state_dict': ckpt} for ckpt in ckpts_delta]
        for j in range(len(ckpt_paths)):
            print(f"Scaling SVD shift by {svd_delta_scales_1d[j]} (1d), {svd_delta_scales_2d[j]} (2d), {svd_delta_scales_4d[j]} (4d)")
            ckpts_delta[j] = scale_svd_shifts(
                ckpt_svd=ckpt_svd,
                ckpt_delta=ckpts_delta[j],
                svd_delta_scale_1d=svd_delta_scales_1d[j],
                svd_delta_scale_2d=svd_delta_scales_2d[j],
                svd_delta_scale_3d=svd_delta_scales_3d[j],
                svd_delta_scale_4d=svd_delta_scales_4d[j],
            )
        if how_to_load == "sequential":
            state_dict = {}
            for j in range(len(ckpt_paths)):
                state_dict.update(ckpts_delta[j]['state_dict'])
        elif how_to_load == "sum":
            state_dict = {}
            for k in ckpts_delta[0]["state_dict"]:
                delta = 0.
                for j in range(len(ckpt_paths)):
                    delta = delta + ckpts_delta[j]["state_dict"][k]
                state_dict[k] = delta
        elif how_to_load == "max":  # element-wise max
            state_dict = {}
            for k in ckpts_delta[-1]["state_dict"]:
                delta = ckpts_delta[-1]["state_dict"][k]
                for j in range(len(ckpt_paths)-1):
                    delta = torch.maximum(delta, ckpts_delta[j]["state_dict"][k])
                state_dict[k] = delta
        elif how_to_load == "max_norm":  # argmax of norm
            state_dict = {}
            for k in ckpts_delta[-1]["state_dict"]:
                delta = ckpts_delta[-1]["state_dict"][k]
                for j in range(len(ckpt_paths)-1):
                    if torch.norm(delta) < torch.norm(ckpts_delta[j]["state_dict"][k]):
                        delta = ckpts_delta[j]["state_dict"][k]
                state_dict[k] = delta
        elif how_to_load == "interpolate":
            weight_interp_alphas = str2float(weight_kwargs.get('weight_interp_alphas', [0.5, 0.5]))
            state_dict = {}
            for k in ckpts_delta[0]["state_dict"]:
                delta = 0.
                for j in range(len(ckpt_paths)):
                    delta = delta + ckpts_delta[j]["state_dict"][k] * weight_interp_alphas[j]
                state_dict[k] = delta
        else:
            raise NotImplementedError
        ckpt_delta = {'state_dict': state_dict}
        print("Converting parametrized weights...")
        pl_sd = convert_svd_shift_to_weight(
            ckpt_svd,
            ckpt_delta,
            svd_delta_scale=1.,  # NOTE: already scaled
            svd_delta_scale_1d=1.,
            svd_delta_scale_2d=1.,
            svd_delta_scale_3d=1.,
            svd_delta_scale_4d=1.,
            svd_kwargs=weight_kwargs,
        )

    else:  # NOTE: load ckpts as full weights
        weight_delta_scales = str2float(weight_kwargs.get('weight_delta_scales', [1.]))
        # NOTE: scale deltas by scales
        ckpt_orig = None
        if how_to_load in ["sum", "max"] or any(np.array(weight_delta_scales) != 1):
            if ckpt_orig_path is None:
                print(f"Using model's state_dict as original weights")
                ckpt_orig = {'state_dict': model.state_dict()}
            else:
                print(f"Loading original model from {ckpt_orig_path}")
                ckpt_orig = torch.load(ckpt_orig_path, map_location="cpu")
                ckpt_orig = ckpt_orig if 'state_dict' in ckpt_orig else {'state_dict': ckpt_orig}
        print(f"Loading model from {ckpt_paths}")
        ckpts = [torch.load(ckpt_path, map_location="cpu") for ckpt_path in ckpt_paths]
        ckpts = [ckpt if 'state_dict' in ckpt else {'state_dict': ckpt} for ckpt in ckpts]
        for j in range(len(ckpt_paths)):
            print(f"Scaling weight delta by {weight_delta_scales[j]}")
            ckpts[j] = scale_weight_deltas(
                ckpt_orig,
                ckpts[j],
                weight_delta_scale=weight_delta_scales[j],
            )
        if how_to_load == "sequential":
            state_dict = {}
            for j in range(len(ckpt_paths)):
                state_dict.update(ckpts[j]['state_dict'])
        elif how_to_load == "sum":
            state_dict = {}
            for k in ckpts[0]["state_dict"]:
                delta = 0.
                for j in range(len(ckpt_paths)):
                    delta = delta + ckpts[j]['state_dict'][k] - ckpt_orig['state_dict'][k]
                state_dict[k] = ckpt_orig['state_dict'][k] + delta
        elif how_to_load == "max":  # element-wise max
            state_dict = {}
            for k in ckpts[-1]["state_dict"]:
                delta = ckpts[-1]['state_dict'][k] - ckpt_orig['state_dict'][k]
                for j in range(len(ckpt_paths)-1):
                    delta = torch.maximum(delta, ckpts[j]['state_dict'][k] - ckpt_orig['state_dict'][k])
                state_dict[k] = ckpt_orig['state_dict'][k] + delta
        elif how_to_load == "max_norm":  # argmax of norm
            state_dict = {}
            for k in ckpts[-1]["state_dict"]:
                delta = ckpts[-1]['state_dict'][k] - ckpt_orig['state_dict'][k]
                for j in range(len(ckpt_paths)-1):
                    if torch.norm(delta) < torch.norm(ckpts[j]['state_dict'][k] - ckpt_orig['state_dict'][k]):
                        delta = ckpts[j]['state_dict'][k] - ckpt_orig['state_dict'][k]
                state_dict[k] = ckpt_orig['state_dict'][k] + delta
        elif how_to_load == "interpolate":
            weight_interp_alphas = str2float(weight_kwargs.get('weight_interp_alphas', [0.5, 0.5]))
            state_dict = {}
            for k in ckpts[0]["state_dict"]:
                weight = 0.
                for j in range(len(ckpt_paths)):
                    weight = weight + ckpts[j]["state_dict"][k] * weight_interp_alphas[j]
                state_dict[k] = weight
        else:
            raise NotImplementedError
        pl_sd = {'state_dict': state_dict}

    sd = pl_sd["state_dict"]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model


""" Low rank
"""
class _LowRank(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        dim: int = 0,
        rank: int = 1,
        alpha: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        self.dim = dim if dim >= 0 else dim + ndim
        self.rank = rank
        self.alpha = alpha
        if ndim > 1:
            # For ndim == 1 no need to do anything
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.shape
            self.rank = min(rank, min(h, w))
            u = torch.randn(h, self.rank).cuda() * 0.0
            vh = torch.randn(self.rank, w).cuda() * 1.0
            self.lora_u = nn.Parameter(u, requires_grad=True)
            self.lora_vh = nn.Parameter(vh, requires_grad=True)
        else:
            self.lora_u = nn.Parameter(torch.randn_like(weight).cuda() * 0.0, requires_grad=True)
            self.lora_vh = torch.ones(1).cuda()

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Precondition
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))

        return weight.flatten(1)  # NOTE: for conv2d it will be (c_out, c_in * k_x * k_y)
    
    def _reshape_matrix_to_weight(self, weight_mat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        if self.dim != 0:
            # dim permuted to front
            weight = weight_mat.reshape(shape[self.dim], *(shape[d] for d in range(len(shape)) if d != self.dim))
            weight = weight.permute(*np.argsort([self.dim] + [d for d in range(weight.dim()) if d != self.dim]))
        else:
            weight = weight_mat.reshape(shape)

        return weight

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            return weight + self.lora_u * self.alpha / self.rank
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            weight_mat_delta = self.lora_u @ self.lora_vh * self.alpha / self.rank
            return self._reshape_matrix_to_weight(weight_mat + weight_mat_delta, weight.shape)


def low_rank(module, name='weight', dim=None, rank=1, svd_kwargs=None):
    if svd_kwargs is None:
        svd_kwargs = {}
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    new_module = _LowRank(weight, dim=dim, rank=rank, **svd_kwargs)
    nn.utils.parametrize.register_parametrization(module, name, new_module)
    return module, new_module  # NOTE: module is parametrized inplace


def register_low_rank(
    model,
    rank=1,
    cached_svd_state_dict=None,
    bias_trainable=True, svd_kwargs=None,
    parametrized_module_list=[],
    new_module_dict={'1d': [], '2d': [], '3d': [], '4d': [], 'bias': []},
    new_module_params={'1d': [], '2d': [], '3d': [], '4d': [], 'bias': []},
    **kwargs
):
    # ==========
    model.cuda()  # NOTE: hardcoded, always move to cuda for SVD computation
    # ==========

    device = next(model.parameters()).device
    if cached_svd_state_dict is not None and 'state_dict' in cached_svd_state_dict:
        cached_svd_state_dict = cached_svd_state_dict['state_dict']
    for n, m in model.named_modules():
        n_ = n+'.weight'
        if is_weight_module(m, n_):
            if cached_svd_state_dict is not None and n + '.parametrizations.weight.0._s' in cached_svd_state_dict:
                cached_svd_params = {
                    'u': cached_svd_state_dict[n + '.parametrizations.weight.0._u'].to(device),
                    's': cached_svd_state_dict[n + '.parametrizations.weight.0._s'].to(device),
                    'vh': cached_svd_state_dict[n + '.parametrizations.weight.0._vh'].to(device),
                    'residual': cached_svd_state_dict[n + '.parametrizations.weight.0._residual'].to(device),
                }
            else:
                cached_svd_params = None
            m, new_m = low_rank(m, 'weight', dim=0, rank=rank,
                svd_kwargs=svd_kwargs)
            parametrized_module_list.append(m)
            new_module_dict[f'{m.weight.ndim}d'].append(new_m)
            new_module_params[f'{m.weight.ndim}d'] += list(new_m.parameters())
            if bias_trainable and hasattr(m, 'bias') and m.bias is not None:
                new_module_params['bias'] += [m.bias]
    return parametrized_module_list, new_module_dict, new_module_params


def scale_lora_deltas(
    ckpt_svd,
    ckpt_delta,
    svd_delta_scale_1d=1.,
    svd_delta_scale_2d=1.,
    svd_delta_scale_3d=1.,
    svd_delta_scale_4d=1.,
):
    state_dict = {}
    names_delta = [k for k in ckpt_delta['state_dict'].keys() if 'parametrizations' in k]
    names_delta = list(set([n.split('parametrizations')[0] for n in names_delta]))
    names_bias = [k for k in ckpt_delta['state_dict'].keys() if k.endswith('.bias')]
    for k in names_delta:
        scale = 1.
        name_original = k + 'parametrizations.weight.original'
        for delta_name_suffix in ['lora_u', 'lora_vh']:
            weight = ckpt_svd['state_dict'][name_original]
            name_delta = k + f'parametrizations.weight.0.{delta_name_suffix}'
            if name_delta in ckpt_delta['state_dict']:
                if weight.ndim == 1:
                    scale *= svd_delta_scale_1d
                elif weight.ndim == 2:
                    scale *= svd_delta_scale_2d
                elif weight.ndim == 3:
                    scale *= svd_delta_scale_3d
                elif weight.ndim == 4:
                    scale *= svd_delta_scale_4d
                state_dict[name_delta] = ckpt_delta['state_dict'][name_delta] * scale
    for k in names_bias:
        bias_delta = ckpt_delta['state_dict'][k] - ckpt_svd['state_dict'][k]
        state_dict[k] = ckpt_svd['state_dict'][k] + svd_delta_scale_1d * bias_delta
    return {'state_dict': state_dict}


def convert_lora_delta_to_weight(
    ckpt_svd,
    ckpt_delta,
    svd_kwargs=None,
):
    """ Combine deltas back to full weights
    """
    lora_rank = 1.  # NOTE: hardcoded, will be removed in future
    lora_alpha = 1.  # NOTE: hardcoded, will be removed in future

    if svd_kwargs is None:
        svd_kwargs = {}
    names_all = [k for k in ckpt_svd['state_dict'].keys() if 'parametrizations' in k]
    names_all = list(set([n.split('parametrizations')[0] for n in names_all]))
    names_delta = [k for k in ckpt_delta['state_dict'].keys() if 'parametrizations' in k]
    names_delta = list(set([n.split('parametrizations')[0] for n in names_delta]))
    names_bias = [k for k in ckpt_delta['state_dict'].keys() if k.endswith('.bias')]
    for k in tqdm(names_delta):
        name = k + 'parametrizations.weight.original'
        weight = ckpt_svd['state_dict'][name]
        device = weight.device
        if weight.ndim > 1:
            delta_u = ckpt_delta['state_dict'][k + 'parametrizations.weight.0.lora_u'].cuda()
            delta_vh = ckpt_delta['state_dict'][k + 'parametrizations.weight.0.lora_vh'].cuda()
            with torch.no_grad():
                weight_mat = reshape_weight_to_matrix(weight, 0)  # NOTE: dim=0 is hardcoded
                weight_mat = delta_u @ delta_vh * lora_alpha / lora_rank + weight_mat.cuda()
                weight = reshape_matrix_to_weight(weight_mat, weight.shape, 0)  # NOTE: dim=0 is hardcoded
            ckpt_svd['state_dict'][k + 'weight'] = weight.to(device)
        else:
            name = k + 'parametrizations.weight.0.lora_u'
            delta = ckpt_delta['state_dict'][name]
            ckpt_svd['state_dict'][k + 'weight'] = delta * lora_alpha / lora_rank + \
                ckpt_svd['state_dict'][k + 'parametrizations.weight.original']
    for k in names_bias:
        ckpt_svd['state_dict'][k] = ckpt_delta['state_dict'][k]
    for k in list(set(names_all) - set(names_delta) - set(names_bias)):
        # NOTE: recover non-parametrized weights to original
        name = k + 'parametrizations.weight.original'
        ckpt_svd['state_dict'][k + 'weight'] = ckpt_svd['state_dict'][name]
        del ckpt_svd['state_dict'][name]
    return ckpt_svd


def load_lora_ckpt(
    model,
    ckpt_paths,
    ckpt_svd_path=None,
    weight_kwargs=None,
    how_to_load="sequential",
    verbose=False,
):
    """
    If how_to_load == 'sequential', state dicts in ckpt_paths will be sequentially
    updated; If how_to_load == 'merge', weight deltas will be merged.
    """
    if isinstance(ckpt_paths, str):
        ckpt_paths = re.split(",|:|;", ckpt_paths)

    svd_delta_scales_1d = str2float(weight_kwargs.get('svd_delta_scales_1d', [1.]))
    svd_delta_scales_2d = str2float(weight_kwargs.get('svd_delta_scales_2d', [1.]))
    svd_delta_scales_3d = str2float(weight_kwargs.get('svd_delta_scales_3d', [1.]))
    svd_delta_scales_4d = str2float(weight_kwargs.get('svd_delta_scales_4d', [1.]))
    # NOTE: scale deltas by scales
    print(f"Loading model from {ckpt_svd_path}")
    ckpt_svd = torch.load(ckpt_svd_path, map_location="cpu")
    if 'state_dict' not in ckpt_svd:
        ckpt_svd = {'state_dict': ckpt_svd}
    print(f"Loading deltas from {ckpt_paths}")
    ckpts_delta = [torch.load(ckpt_path, map_location="cpu") for ckpt_path in ckpt_paths]
    ckpts_delta = [ckpt if 'state_dict' in ckpt else {'state_dict': ckpt} for ckpt in ckpts_delta]
    for j in range(len(ckpt_paths)):
        print(f"Scaling LoRA delta by {svd_delta_scales_1d[j]} (1d), {svd_delta_scales_2d[j]} (2d), {svd_delta_scales_4d[j]} (4d)")
        ckpts_delta[j] = scale_lora_deltas(
            ckpt_svd=ckpt_svd,
            ckpt_delta=ckpts_delta[j],
            svd_delta_scale_1d=svd_delta_scales_1d[j],
            svd_delta_scale_2d=svd_delta_scales_2d[j],
            svd_delta_scale_3d=svd_delta_scales_3d[j],
            svd_delta_scale_4d=svd_delta_scales_4d[j],
        )
    if how_to_load == "sequential":
        state_dict = {}
        for j in range(len(ckpt_paths)):
            state_dict.update(ckpts_delta[j]['state_dict'])
    elif how_to_load == "sum":
        state_dict = {}
        for k in ckpts_delta[0]["state_dict"]:
            delta = 0.
            for j in range(len(ckpt_paths)):
                delta = delta + ckpts_delta[j]["state_dict"][k]
            state_dict[k] = delta
    elif how_to_load == "max":  # element-wise max
        state_dict = {}
        for k in ckpts_delta[-1]["state_dict"]:
            delta = ckpts_delta[-1]["state_dict"][k]
            for j in range(len(ckpt_paths)-1):
                delta = torch.maximum(delta, ckpts_delta[j]["state_dict"][k])
            state_dict[k] = delta
    elif how_to_load == "max_norm":  # argmax of norm
        state_dict = {}
        for k in ckpts_delta[-1]["state_dict"]:
            delta = ckpts_delta[-1]["state_dict"][k]
            for j in range(len(ckpt_paths)-1):
                if torch.norm(delta) < torch.norm(ckpts_delta[j]["state_dict"][k]):
                    delta = ckpts_delta[j]["state_dict"][k]
            state_dict[k] = delta
    elif how_to_load == "interpolate":
        weight_interp_alphas = str2float(weight_kwargs.get('weight_interp_alphas', [0.5, 0.5]))
        state_dict = {}
        for k in ckpts_delta[0]["state_dict"]:
            delta = 0.
            for j in range(len(ckpt_paths)):
                delta = delta + ckpts_delta[j]["state_dict"][k] * weight_interp_alphas[j]
            state_dict[k] = delta
    else:
        raise NotImplementedError
    ckpt_delta = {'state_dict': state_dict}
    print("Converting parametrized weights...")
    pl_sd = convert_lora_delta_to_weight(
        ckpt_svd,
        ckpt_delta,
        svd_kwargs=weight_kwargs,
    )

    sd = pl_sd["state_dict"]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model


def get_lora_delta_state_dict(model):
    bn_state_dict = {}
    state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if 'lora_' in k:
            new_state_dict[k] = state_dict[k]
        if 'running' in k:
            bn_state_dict[k] = state_dict[k]
    return new_state_dict, bn_state_dict


def convert_ckpt_to_svd(model):
    ckpt = {'state_dict': model.state_dict()}
    new_state_dict = {}
    name_norms = []
    large_name_norms = []
    # placeholder = torch.zeros(1)
    for k in tqdm(ckpt['state_dict'].keys()):
        if k.endswith('.weight'):
            weight = ckpt['state_dict'][k]
            device = weight.device
            name = k[:-6]
            if weight.ndim > 1:
                weight_shape = weight.shape
                weight_mat = reshape_weight_to_matrix(weight.cuda(), 0)  # NOTE: always to cuda
                u, s, vh = torch.linalg.svd(weight_mat, full_matrices=False)
                residual = weight_mat - u @ torch.diag(s) @ vh
                residual_norm = torch.norm(residual).item() / residual.numel()
                # print(torch.norm(residual))
                name_norms.append((name, residual_norm))
                if residual_norm > 1e-6:
                    print("residual is not zero")
                    large_name_norms.append((name, residual_norm))
                new_state_dict[name + 'parametrizations.weight.0._u'] = u.to(device)
                new_state_dict[name + 'parametrizations.weight.0._s'] = s.to(device)
                new_state_dict[name + 'parametrizations.weight.0._vh'] = vh.to(device)
                new_state_dict[name + 'parametrizations.weight.0._residual'] = residual.to(device)
                new_state_dict[name + 'parametrizations.weight.original'] = weight
                # new_state_dict[name + 'parametrizations.weight.0.eigenvalues_delta'] = torch.zeros_like(s)
            else:
                new_state_dict[name + 'parametrizations.weight.original'] = weight
                # new_state_dict[name + 'parametrizations.weight.0.eigenvalues_delta'] = torch.zeros_like(weight)
        else:
            new_state_dict[k] = ckpt['state_dict'][k]
    return new_state_dict
