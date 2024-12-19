import torch
import torch.nn as nn
from source.layers.klayer import (
    KLayer,
)
from source.layers.common_layers import (
    ReadOutConv,
    BNReLUConv2d,
    FF,
    ResBlock,
)
from source.layers.common_fns import positionalencoding2d


from source.data.datasets.sudoku.sudoku import convert_onehot_to_int


class SudokuAKOrN(nn.Module):

    def __init__(
        self,
        n,
        ch=64,
        L=1,
        T=16,
        gamma=1.0,
        J="attn",
        use_omega=True,
        global_omg=True,
        init_omg=0.1,
        learn_omg=False,
        nl=True,
        heads=8,
    ):
        super().__init__()
        self.n = n
        self.L = L
        self.ch = ch
        self.embedding = nn.Embedding(10, ch)

        hw = [9, 9]

        self.layers = nn.ModuleList()
        for l in range(self.L):
            self.layers.append(
                nn.ModuleList(
                    [
                        KLayer(
                            n,
                            ch,
                            J,
                            use_omega=use_omega,
                            c_norm=None,
                            hw=hw,
                            global_omg=global_omg,
                            init_omg=init_omg,
                            heads=heads,
                            learn_omg=learn_omg,
                            gta=True,
                        ),
                        nn.Sequential(
                            ReadOutConv(ch, ch, n, 1, 1, 0),
                            ResBlock(FF(ch, ch, ch, 1, 1, 0)) if nl else nn.Identity(),
                            BNReLUConv2d(ch, ch, 1, 1, 0) if nl else nn.Identity(),
                        ),
                    ]
                )
            )

        self.out = nn.Sequential(nn.ReLU(), nn.Conv2d(ch, 9, 1, 1, 0))

        self.T = T
        self.gamma = torch.nn.Parameter(torch.Tensor([gamma]))
        self.fixed_noise = False
        self.x0 = nn.Parameter(torch.randn(1, ch, 9, 9))

    def feature(self, inp, is_input):
        # inp: torch.Tensor of shape [B, 9, 9, 9] the last dim repreents the digit in the one-hot representation.
        inp = convert_onehot_to_int(inp)
        c = self.embedding(inp).permute(0, 3, 1, 2)
        is_input = is_input.permute(0, 3, 1, 2)
        xs = []
        es = []

        # generate random oscillatores
        if self.fixed_noise:
            n = torch.randn(
                *(c.shape), generator=torch.Generator(device="cpu").manual_seed(42)
            ).to(c.device)
            x = is_input * c + (1 - is_input) * n
        else:
            n = torch.randn_like(c)
            x = is_input * c + (1 - is_input) * n

        for _, (klayer, readout) in enumerate(self.layers):
            # Process x and c.
            _xs, _es = klayer(
                x,
                c,
                self.T,
                self.gamma,
            )
            xs.append(_xs)
            es.append(_es)
            c = readout(_xs[-1])
        return c, xs, es

    def forward(self, c, is_input, return_xs=False, return_es=False):
        out, xs, es = self.feature(c, is_input)
        out = self.out(out).permute(0, 2, 3, 1)
        ret = [out]
        if return_xs:
            ret.append(xs)
        if return_es:
            ret.append(es)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
