# https://github.com/JCruan519/VM-UNet/blob/main/models/vmunet/vmamba.py
import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

# Conditional version of the original SS2D
class ConditionalSS2D(SS2D):
    """
    Difference from original SS2D:
        original:
            x -> u, B, C, Delta, z

        conditional:
            target      -> u
            param_source -> B, C, Delta
            z_source     -> z

    Input/output format is BHWC, same as the original SS2D in this vmamba.py.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            dropout=dropout,
            conv_bias=conv_bias,
            bias=bias,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        factory_kwargs = {"device": device, "dtype": dtype}

        # Replace the original single in_proj with three explicit projections.
        # target -> u
        self.u_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # param_source -> B / C / Delta
        self.param_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # z_source -> multiplicative branch
        self.z_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        if hasattr(self, "in_proj"):
            del self.in_proj


    def _cross_scan_4dir(self, x: torch.Tensor):
        """
        Same 4-direction scan as original SS2D.forward_corev0.

        Input:
            x: [B, D, H, W]

        Output:
            xs: [B, K, D, L], K=4
        """
        B, D, H, W = x.shape
        L = H * W
        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        return xs

    def forward_core_conditional(self, u: torch.Tensor, param: torch.Tensor):
        """
        u:
            target spatial branch after projection + DWConv + SiLU, [B, D, H, W]

        param:
            cross-modal frequency low/high branch after projection + SiLU, [B, D, H, W]
        """
        self.selective_scan = selective_scan_fn

        B, D, H, W = u.shape
        L = H * W
        K = 4

        # 1. target provides selective scan input u
        us = self._cross_scan_4dir(u)          # [B, K, D, L]

        # 2. condition provides B / C / Delta
        ps = self._cross_scan_4dir(param)      # [B, K, D, L]

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            ps.view(B, K, -1, L),
            self.x_proj_weight,
        )

        dts, Bs, Cs = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )

        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(B, K, -1, L),
            self.dt_projs_weight,
        )

        us = us.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            us,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(
            out_y[:, 1].view(B, -1, W, H),
            dim0=2,
            dim1=3,
        ).contiguous().view(B, -1, L)

        invwh_y = torch.transpose(
            inv_y[:, 1].view(B, -1, W, H),
            dim0=2,
            dim1=3,
        ).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(
        self,
        target: torch.Tensor,
        param_source: torch.Tensor,
        z_source: Optional[torch.Tensor] = None,
    ):
        """
        Input format:
            target:       [B, H, W, C]
            param_source: [B, H, W, C]
            z_source:     [B, H, W, C]

        If z_source is None, param_source is also used for the multiplicative branch.
        """
        if z_source is None:
            z_source = param_source

        B, H, W, C = target.shape

        # target spatial feature -> u
        u = self.u_proj(target)
        u = u.permute(0, 3, 1, 2).contiguous()
        u = self.act(self.conv2d(u))

        # cross-modal low/high feature -> B/C/Delta
        param = self.param_proj(param_source)
        param = param.permute(0, 3, 1, 2).contiguous()
        param = self.act(param)

        # cross-modal low/high feature -> multiplicative branch z
        z = self.z_proj(z_source)

        y1, y2, y3, y4 = self.forward_core_conditional(u, param)
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        # same multiplicative branch style as original SS2D
        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# Wrapper of the original VSSBlock. 
# This wrapper allows us to use BCHW feature maps.
class VSSBlockBCHW(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        d_state: int = 16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.block = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCHW -> BHWC
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.block(x)
        # BHWC -> BCHW
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

# Conditional VSS block based on the original VSSBlock.
class ConditionalVSSBlock(nn.Module):
    """
    Conditional VSS block based on the original VSSBlock.

    Original VSSBlock:
        x = x + DropPath(SS2D(LN(x)))

    Conditional version:
        target = target + DropPath(
            ConditionalSS2D(
                LN(target),
                LN(param_source),
                LN(z_source)
            )
        )
    """
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()

        self.ln_target = norm_layer(hidden_dim)
        self.ln_param = norm_layer(hidden_dim)
        self.ln_z = norm_layer(hidden_dim)

        self.self_attention = ConditionalSS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )

        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        target: torch.Tensor,
        param_source: torch.Tensor,
        z_source: Optional[torch.Tensor] = None,
    ):
        if z_source is None:
            z_source = param_source

        out = self.self_attention(
            self.ln_target(target),
            self.ln_param(param_source),
            self.ln_z(z_source),
        )

        return target + self.drop_path(out)


# BCHW wrapper of ConditionalVSSBlock.
class ConditionalVSSBlockBCHW(nn.Module):
    """
    External input:
        target:       [B, C, H, W]
        param_source: [B, C, H, W]
        z_source:     [B, C, H, W]

    Internal VMamba format:
        [B, H, W, C]
    """
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        d_state: int = 16,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.block = ConditionalVSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )

    def forward(
        self,
        target: torch.Tensor,
        param_source: torch.Tensor,
        z_source: Optional[torch.Tensor] = None,
    ):
        target_bhwc = target.permute(0, 2, 3, 1).contiguous()
        param_bhwc = param_source.permute(0, 2, 3, 1).contiguous()

        if z_source is None:
            z_bhwc = param_bhwc
        else:
            z_bhwc = z_source.permute(0, 2, 3, 1).contiguous()

        out = self.block(target_bhwc, param_bhwc, z_bhwc)

        return out.permute(0, 3, 1, 2).contiguous()







# ===============================
# Frequency-domain SS2D scan paths
# ===============================

def _spiral_clockwise_indices(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    第一种扫描：正螺旋扫描。

    从外圈到内圈，顺时针矩形螺旋。
    从左上角开始：
        向右 -> 向下 -> 向左 -> 向上 -> 继续内圈

    返回:
        idx: [H*W]
        row-major flatten 后的索引。
    """
    top, bottom = 0, H - 1
    left, right = 0, W - 1
    order = []

    while top <= bottom and left <= right:
        # top row: left -> right
        for c in range(left, right + 1):
            order.append(top * W + c)
        top += 1

        # right col: top -> bottom
        for r in range(top, bottom + 1):
            order.append(r * W + right)
        right -= 1

        # bottom row: right -> left
        if top <= bottom:
            for c in range(right, left - 1, -1):
                order.append(bottom * W + c)
            bottom -= 1

        # left col: bottom -> top
        if left <= right:
            for r in range(bottom, top - 1, -1):
                order.append(r * W + left)
            left += 1

    return torch.tensor(order, dtype=torch.long, device=device)


def _spiral_counterclockwise_indices(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    第二种扫描：第一种正螺旋扫描的逆向扫描。

    注意：
        这里不是重新写一个逆时针螺旋，
        而是严格取第一种扫描序列的反向。

    返回:
        idx: [H*W]
    """
    spiral_cw = _spiral_clockwise_indices(H, W, device)
    spiral_reverse = torch.flip(spiral_cw, dims=[0])
    return spiral_reverse


def _bresenham_line(r0: int, c0: int, r1: int, c1: int):
    """
    Bresenham 直线算法。

    从 (r0, c0) 到 (r1, c1)，返回线上经过的所有整数网格点。
    点格式:
        (row, col)
    """
    points = []

    dx = abs(c1 - c0)
    dy = -abs(r1 - r0)

    sx = 1 if c0 < c1 else -1
    sy = 1 if r0 < r1 else -1

    err = dx + dy

    r, c = r0, c0

    while True:
        points.append((r, c))

        if r == r1 and c == c1:
            break

        e2 = 2 * err

        if e2 >= dy:
            err += dy
            c += sx

        if e2 <= dx:
            err += dx
            r += sy

    return points


def _boundary_points_from_topleft_clockwise(H: int, W: int):
    """
    边界点顺序。

    从左上角开始：
        1. 第一行：左 -> 右
        2. 最右列：上 -> 下
        3. 最后一行：右 -> 左
        4. 最左列：下 -> 上

    这样就对应：
        先从中心扫向左上角，
        再扫向左上角右边一格，
        然后依次沿边界扫。
    """
    boundary = []

    # top row: left -> right
    for c in range(W):
        boundary.append((0, c))

    # right col: top+1 -> bottom
    for r in range(1, H):
        boundary.append((r, W - 1))

    # bottom row: right-1 -> left
    if H > 1:
        for c in range(W - 2, -1, -1):
            boundary.append((H - 1, c))

    # left col: bottom-1 -> top+1
    if W > 1:
        for r in range(H - 2, 0, -1):
            boundary.append((r, 0))

    return boundary


def _center_out_indices(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    第三种扫描：中心向四周扫描。

    规则：
        1. 以 fftshift 后的频谱中心为起点。
           对偶数尺寸，例如 8x8，中心取 (H//2, W//2)，即 (4, 4)。

        2. 先从中心用 Bresenham 线扫向左上角。

        3. 然后从中心扫向左上角右边一格，
           再依次沿边界点扫描。

        4. 每条 Bresenham 线上经过的点都加入扫描序列。

        5. 已经扫描过的位置不重复加入。

    返回:
        idx: [H*W]
    """
    center_r = H // 2
    center_c = W // 2

    boundary_points = _boundary_points_from_topleft_clockwise(H, W)

    order = []
    visited = set()

    for end_r, end_c in boundary_points:
        line_points = _bresenham_line(center_r, center_c, end_r, end_c)

        for r, c in line_points:
            pos = r * W + c

            if pos not in visited:
                visited.add(pos)
                order.append(pos)

    # 理论上：中心到所有边界点的 Bresenham 线可以覆盖全部位置。
    # 这里加兜底，保证绝对不会漏扫。
    if len(order) != H * W:
        missing = []

        for r in range(H):
            for c in range(W):
                pos = r * W + c

                if pos not in visited:
                    dr = r - center_r
                    dc = c - center_c
                    dist2 = dr * dr + dc * dc
                    missing.append((dist2, r, c, pos))

        # 缺失点按照离中心由近到远补上
        missing.sort(key=lambda x: (x[0], x[1], x[2]))

        for _, _, _, pos in missing:
            order.append(pos)

    return torch.tensor(order, dtype=torch.long, device=device)


def _frequency_scan_indices(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    频域编码器专用的四种扫描：
        1. 正螺旋扫描
        2. 正螺旋扫描的逆向扫描
        3. 中心向四周 Bresenham 线扫描
        4. 第三种扫描的逆向扫描，也就是四周向中心

    返回:
        idx: [4, H*W]

    注意:
        每一行都是 0 ~ H*W-1 的完整排列。
        保证每个像素点都会被扫描到一次。
    """
    spiral_cw = _spiral_clockwise_indices(H, W, device)

    # 第二种：第一种的逆向扫描
    spiral_reverse = torch.flip(spiral_cw, dims=[0])

    # 第三种：中心 -> 四周，Bresenham 线扫描
    center_out = _center_out_indices(H, W, device)

    # 第四种：第三种的逆向扫描，四周 -> 中心
    surround_in = torch.flip(center_out, dims=[0])

    idx = torch.stack(
        [
            spiral_cw,
            spiral_reverse,
            center_out,
            surround_in,
        ],
        dim=0,
    )

    return idx


def _check_scan_indices(idx: torch.Tensor, L: int):
    """
    调试用：检查四种扫描是否都覆盖全部位置。
    正常训练时可以不开。
    """
    if idx.shape != (4, L):
        raise RuntimeError(
            f"frequency scan index shape should be (4, {L}), got {tuple(idx.shape)}"
        )

    for k in range(4):
        unique_num = torch.unique(idx[k]).numel()

        if unique_num != L:
            raise RuntimeError(
                f"frequency scan path {k} does not cover every position exactly once. "
                f"unique={unique_num}, expected={L}"
            )

        if idx[k].min().item() != 0 or idx[k].max().item() != L - 1:
            raise RuntimeError(
                f"frequency scan path {k} has invalid index range: "
                f"min={idx[k].min().item()}, max={idx[k].max().item()}, expected=[0, {L - 1}]"
            )


class FrequencySS2D(SS2D):
    """
    频域编码器专用 SS2D。

    和原始 SS2D 的区别：
        原始 SS2D:
            横向正向 + 纵向正向 + 横向反向 + 纵向反向

        FrequencySS2D:
            正螺旋 + 反螺旋 + 中心向四周 + 四周向中心
    """

    def __init__(self, *args, check_scan: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_scan = check_scan

        # 替换原始 SS2D 的 forward_core
        self.forward_core = self.forward_core_frequency

    def _scan_expand(self, x: torch.Tensor):
        """
        输入:
            x: [B, D, H, W]

        输出:
            xs:  [B, 4, D, L]
            idx: [4, L]
        """
        B, D, H, W = x.shape
        L = H * W

        idx = _frequency_scan_indices(H, W, x.device)

        if self.check_scan:
            _check_scan_indices(idx, L)

        flat = x.contiguous().view(B, D, L)

        xs = torch.stack(
            [
                torch.index_select(flat, dim=2, index=idx[k])
                for k in range(4)
            ],
            dim=1,
        )

        return xs, idx

    @staticmethod
    def _scan_merge(out_y: torch.Tensor, idx: torch.Tensor):
        """
        把每个扫描方向的一维序列恢复到原始二维位置。

        输入:
            out_y: [B, 4, D, L]
            idx:   [4, L]

        输出:
            y1, y2, y3, y4，每个都是 [B, D, L]
        """
        B, K, D, L = out_y.shape
        restored = []

        for k in range(K):
            y = out_y.new_zeros(B, D, L)
            index = idx[k].view(1, 1, L).expand(B, D, L)
            y.scatter_(dim=2, index=index, src=out_y[:, k])
            restored.append(y)

        return restored

    def forward_core_frequency(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        # 1. 按频域四种路径展开
        xs, scan_idx = self._scan_expand(x)  # [B, 4, D, L]

        # 2. 和原始 SS2D 一样，生成 delta / B / C
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.view(B, K, -1, L),
            self.x_proj_weight,
        )

        dts, Bs, Cs = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )

        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(B, K, -1, L),
            self.dt_projs_weight,
        )

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # 3. Selective Scan
        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        # 4. 按每种扫描的索引恢复回原始位置
        y1, y2, y3, y4 = self._scan_merge(out_y, scan_idx)

        return y1, y2, y3, y4


class FrequencyVSSBlock(nn.Module):
    """
    频域编码器专用 VSS Block。
    内部使用 FrequencySS2D。
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = FrequencySS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class FrequencyVSSBlockBCHW(nn.Module):
    """
    BCHW 版本的频域 VSS Block。

    外部输入:
        [B, C, H, W]

    内部转换成 VMamba 格式:
        [B, H, W, C]
    """

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        d_state: int = 16,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.block = FrequencyVSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCHW -> BHWC
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.block(x)

        # BHWC -> BCHW
        x = x.permute(0, 3, 1, 2).contiguous()

        return x
