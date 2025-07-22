import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

import numpy as np

from source.layers.common_layers import (
    ScaleAndBias,
    Attention,
)

from source.layers.kutils import (
    reshape,
    reshape_back,
    normalize,
)

from source.models.classification.analysis_utils import (
    reshape_blocks_to_tensor, reshape_tensor_to_blocks
)

from einops.layers.torch import Rearrange


class OmegaLayer(nn.Module):

    def __init__(self, n, ch, init_omg=0.1, global_omg=False, learn_omg=True):
        super().__init__()
        self.n = n
        self.ch = ch
        self.global_omg = global_omg

        if not learn_omg:
            print("Not learning omega")

        if n % 2 != 0:
            # n is odd
            raise NotImplementedError
        else:
            # n is even
            if global_omg:
                self.omg_param = nn.Parameter(
                    init_omg * (1 / np.sqrt(2)) * torch.ones(2), requires_grad=learn_omg
                )
            else:
                self.omg_param = nn.Parameter(
                    init_omg * (1 / np.sqrt(2)) * torch.ones(ch // 2, 2),
                    requires_grad=learn_omg,
                )

    def forward(self, x):
        _x = reshape(x, 2)
        if self.global_omg:
            omg = torch.linalg.norm(self.omg_param).repeat(_x.shape[1])
        else:
            omg = torch.linalg.norm(self.omg_param, dim=1)
        omg = omg[None]
        for _ in range(_x.ndim - 3):
            omg = omg.unsqueeze(-1)
        omg_x = torch.stack([omg * _x[:, :, 1], -omg * _x[:, :, 0]], dim=2)
        omg_x = reshape_back(omg_x)
        return omg_x


class KLayer(nn.Module):  # Kuramoto layer

    def __init__(
        self,
        n,
        ch,
        J="conv",
        J_bias=False,      # added by SK on Jul 4, 2025
        c_norm="gn",
        use_omega=True,
        init_omg=1.0,
        ksize=3,
        gta=False,
        hw=None,
        global_omg=False,
        heads=8,
        learn_omg=True,
        apply_proj=True,
        bp_steps=None,   #added by SK on Jul 16, 2025
    ):
        # connnectivity is either 'conv' or 'ca'
        super().__init__()
        assert (ch % n) == 0
        self.n = n
        self.ch = ch
        self.use_omega  = use_omega
        self.global_omg = global_omg
        self.apply_proj = apply_proj
        self.bp_steps   = bp_steps

        self.omg = (
            OmegaLayer(n, ch, init_omg, global_omg, learn_omg)
            if self.use_omega
            else nn.Identity()
        )

        if J == "conv":
            self.connectivity = nn.Conv2d(ch, ch, ksize, 1, ksize // 2, bias=J_bias)
            self.x_type = "image"
        elif J == "conv_repeated_const":    # Added by SK on Jul 4 2025
            self.connectivity = RepeatedConv2d(n, ch, ch, ksize, 1, ksize//2, bias=J_bias)
        elif J == "conv_kuramoto_sakaguchi":  # Added by SK on Jul 21 2025
            self.connectivity = KuramotoSakaguchiConv2d(n, ch, ch, ksize, 1, ksize//2, bias=J_bias)
        elif J == "attn":
            self.connectivity = Attention(
                ch,
                heads=heads,
                weight="conv",
                kernel_size=1,
                stride=1,
                padding=0,
                gta=gta,
                hw=hw,
            )
            self.x_type = "image"
        else:
            raise NotImplementedError

        if c_norm == "gn":
            self.c_norm = nn.GroupNorm(ch // n, ch, affine=True)
        elif c_norm == "sandb":
            self.c_norm = ScaleAndBias(ch, token_input=False)
        elif c_norm is None or c_norm == "none":
            self.c_norm = nn.Identity()
        else:
            raise NotImplementedError

    def project(self, y, x):
        sim = x * y  # similarity between update and current state
        yxx = torch.sum(sim, 2, keepdim=True) * x
        return y - yxx, sim

    def kupdate(self, x: torch.Tensor, c: torch.Tensor = None):
        # compute  \sum_j[J_ij x_j]
        _y = self.connectivity(x)
        # add bias c.
        y = _y + c

        if hasattr(self, "omg"):
            omg_x = self.omg(x)
        else:
            omg_x = torch.zeros_like(x)

        y = reshape(y, self.n)
        x = reshape(x, self.n)

        # project y onto the tangent space
        if self.apply_proj:
            y_yxx, sim = self.project(y, x)
        else:
            y_yxx = y
            sim = y * x

        dxdt = omg_x + reshape_back(y_yxx)
        sim = reshape_back(sim)

        return dxdt, sim

    def forward(self, x: torch.Tensor, c: torch.Tensor, T: int, gamma):#, bp_steps: int = None):
        # x.shape = c.shape = [B, C,...] or [B, T, C]
        xs, es = [], []
        c = self.c_norm(c)
        x = normalize(x, self.n)
        es.append(torch.zeros(x.shape[0]).to(x.device))

        bp_steps = T if self.bp_steps is None else min(self.bp_steps, T)

        # Iterate kuramoto update with condition c
        for t in range(T):
            if t < T - bp_steps:
                with torch.no_grad():
                    dxdt, _sim = self.kupdate(x, c)
                    x = normalize(x + gamma * dxdt, self.n)
                x = x.detach()
            else:
                dxdt, _sim = self.kupdate(x, c)
                x = normalize(x + gamma * dxdt, self.n)
            xs.append(x)
            es.append((-_sim).reshape(x.shape[0], -1).sum(-1))

        return xs, es

################################################################################
#################### BELOW ADDED BY SK 
################################################################################

class RepeatedConv2d(nn.Module):
    """
    conv.weight[o, i, h, w] は
        J[o%2, i%2]   (同じ値を H,W 方向にコピー)
    となる畳み込み層．
    """
    def __init__(self, n=2, ch_in=128, ch_out=128, ksize=9, stride=1, padding=None, bias=False):
        super().__init__()
        assert ch_in % n == 0 and ch_out % n == 0, "チャネル数は 2 の倍数に"
        self.n = n
        self.ch_in  = ch_in
        self.ch_out = ch_out
        self.ksize  = ksize
        self.stride = stride
        self.padding = ksize // 2 if padding is None else padding
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(ch_out))
        else:
            self.register_parameter("bias", None)   # state_dict にも載らない

        # 4 個だけ学習するパラメータ (a,b;c,d)
        self.J = nn.Parameter(torch.randn(n,n) * 0.01)

    def _build_weight(self):
        """
        (2,2) → (2,2,1,1) → repeat → (ch_out, ch_in, ksize, ksize)
        """
        h_rep, w_rep = self.ksize, self.ksize
        oc_rep, ic_rep = self.ch_out // self.n, self.ch_in // self.n
        return self.J.unsqueeze(-1).unsqueeze(-1).repeat(
            oc_rep, ic_rep, h_rep, w_rep)          # 実際に畳み込むフルカーネル

    def forward(self, x):
        w = self._build_weight()
        return nn.functional.conv2d(x, w, bias=self.bias,
                        stride=self.stride,
                        padding=self.padding)
    

class KuramotoSakaguchiConv2d(nn.Module):
    """
    Kuramoto-Sakaguchi 形式の畳み込み層。
    conv.weight[o, i, h, w] は
        J[o%2, i%2]   (同じ値を H,W 方向にコピー)
    となる畳み込み層。そして各J_{ij}は
        J_{ij} = c_{ij} [[ cos(alpha_{ij}), sin(alpha_{ij}) ], [-sin(alpha_{ij}), cos(alpha_{ij}) ]]
    となる！
    """

    def __init__(self, n=2, ch_in=128, ch_out=128, ksize=9, stride=1, padding=None, bias=False):
        super().__init__()
        assert ch_in % n == 0 and ch_out % n == 0, "チャネル数は 2 の倍数に"
        self.n = n
        self.ch_in  = ch_in
        self.ch_out = ch_out
        self.ksize  = ksize
        self.stride = stride
        self.padding = ksize // 2 if padding is None else padding
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(ch_out))
        else:
            self.register_parameter("bias", None)

        #　各J_{ij}は[[a, b], [-b, a]] という形で表現される
        self.J_00 = nn.Parameter(torch.randn(ch_out//n * ch_in//n * ksize * ksize, ) * 0.01)
        self.J_01 = nn.Parameter(torch.randn(ch_out//n * ch_in//n * ksize * ksize, ) * 0.01)


    def _build_weight(self):
        """
        (:,:,h,w) は 2x2行列で
        J_{ij} = c_{ij} [[ cos(alpha_{ij}), sin(alpha_{ij}) ], [-sin(alpha_{ij}), cos(alpha_{ij}) ]]
        となる
        """
        J_blocks = torch.stack([
            torch.stack([ self.J_00, self.J_01], dim=1),  # J_{00}, J_{01}
            torch.stack([-self.J_01, self.J_00], dim=1)   # J_{10}, J_{11}
        ], dim=2)
        w = reshape_blocks_to_tensor(J_blocks, self.n, self.ch_out, self.ch_in, self.ksize, self.ksize)
        return w

    def forward(self, x):
        w = self._build_weight()
        return nn.functional.conv2d(x, w, bias=self.bias,
                        stride=self.stride,
                        padding=self.padding)


class ReadoutOnlyKLayer(nn.Module):
    """
    Control version of KLayer that skips AKOrN iterations but keeps the structure for readout.
    This bypasses the Kuramoto oscillator dynamics (T iterations) but maintains the same
    input/output interface as the original KLayer.
    """
    
    def __init__(
        self,
        n,
        ch,
        c_norm="gn",
    ):
        super().__init__()
        assert (ch % n) == 0
        self.n = n
        self.ch = ch
        
        # Keep the same normalization as original for readout compatibility
        if c_norm == "gn":
            self.c_norm = nn.GroupNorm(ch // n, ch, affine=True)
        elif c_norm == "sandb":
            from source.layers.common_layers import ScaleAndBias
            self.c_norm = ScaleAndBias(ch, token_input=False)
        elif c_norm is None or c_norm == "none":
            self.c_norm = nn.Identity()
        else:
            raise NotImplementedError
            
        # Simple linear transformation to replace Kuramoto dynamics
        # This maintains the same input/output dimensions but removes oscillator iterations
        self.control_transform = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Control forward pass: Skip AKOrN iterations, apply simple transformation.
        
        Returns the same structure as KLayer (xs, es) but without oscillator dynamics:
        - xs: List of states (here just the transformed input repeated T times)
        - es: List of energies (here just zeros)
        """
        xs, es = [], []
        c = self.c_norm(c)
        
        # Apply control transformation instead of Kuramoto iterations
        x_transformed = self.control_transform(x)
        
        # Return the same structure as original KLayer
        # Repeat the transformed state T times to match expected output structure
        # for t in range(T):
        xs.append(x_transformed)
        es.append(torch.zeros(x_transformed.shape[0]).to(x_transformed.device))
            
        return xs, es