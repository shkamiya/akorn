import torch
import torch.nn as nn

from source.layers.klayer import KLayer, ReadoutOnlyKLayer
from source.layers.common_layers import (
    RGBNormalize,
    ReadOutConv,
    Reshape,
    ResBlock,
    FF,
    BNReLUConv2d,
)
from source.layers.kutils import (
    normalize,
)
from source.layers.common_fns import positionalencoding2d
from torchvision.models.resnet import BasicBlock

class MyAKOrN(nn.Module):
    """
    Artficial Kuramoto Oscillator Neurons (AKOrN) for classification tasks.
    
    Args:
        n: Number of nodes per layer (int or list)
        ch: Base number of channels
        out_classes: Number of output classes
        L: Number of layers
        T: Number of time steps (int or list)
        J: Connectivity type (str or list)
        J_bias: convolution bias (t or f), added by SK on Jul 4, 2025       
        ksizes: Kernel sizes (int or list)
        ro_ksize: Readout kernel size
        ro_N: Readout N parameter (int or list)
        norm: Normalization type
        c_norm: normalization type for condtional stimuli (c in the paper)
        gamma: Gamma parameter
        use_omega: Whether to use omega
        init_omg: Initial omega value
        global_omg: Whether to use global omega
        learn_omg: Whether to learn omega
        ensemble: Ensemble size
    """

    def __init__(
        self,
        n=4,
        ch=64,
        out_classes=10,
        L=3,
        T=3,
        J="conv",
        J_bias=False,
        ksizes=[9, 7, 5],
        ro_ksize=3,
        ro_N=2,
        norm="bn",
        c_norm="gn",
        gamma=1.0,
        use_omega=True,
        init_omg=1.0,
        global_omg=True,
        learn_omg=True,
        ensemble=1,
        bp_steps=None,
        ro_only=False,
        ro_fcn="full",
    ):
        super().__init__()
        
        self.fixed_noise = False
        self.n = n
        self.L = L
        self.ensemble = ensemble
        self.gamma = nn.Parameter(torch.tensor([gamma]), requires_grad=False)
        
        # Expand parameters to match number of layers
        self.ns = self._expand_param(n, L)
        self.T = self._expand_param(T, L)
        self.bp_steps = self._expand_param(bp_steps,L)
        self.ro_only = self._expand_param(ro_only, L)
        self.ro_fcn = self._expand_param(ro_fcn, L)

        self.c_norm = c_norm
        self.ch = ch

        J = self._expand_param(J, L)
        ksizes = self._expand_param(ksizes, L)
        ro_N = self._expand_param(ro_N, L)
                
        # Calculate layer dimensions
        strides = [2, 2, 2] + [1] * (L - 3)
        channels = [ch * (2**i) for i in range(L)]
        hw_sizes = [[32 // (2**i), 32 // (2**i)] if i < 3 else [8, 8] for i in range(L)]
        
        # Build network components
        self.rgb_normalize = RGBNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.conv0 = self._create_initial_conv(channels[0])
        self.layers = self._create_layers(L, channels, strides, hw_sizes, J, J_bias, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Reshape(-1, channels[-1]))
        self.out = nn.Linear(channels[-1], out_classes)
        
        # Create normalization layer for readout-only mode to avoid device mismatch
        if c_norm == "gn":
            self.ro_c_norm = nn.GroupNorm(ch // n, ch, affine=True)
        elif c_norm == "sandb":
            from source.layers.common_layers import ScaleAndBias
            self.ro_c_norm = ScaleAndBias(ch, token_input=False)
        elif c_norm is None or c_norm == "none":
            self.ro_c_norm = nn.Identity()
        else:
            raise NotImplementedError
    
    def _expand_param(self, param, length):
        """Expand parameter to match the number of layers."""
        if isinstance(param, (list, tuple)):
            return list(param) if len(param) == length else [param[0]] * length
        return [param] * length
    
    def _create_initial_conv(self, out_channels):
        """Create the initial convolutional layer."""
        return nn.Conv2d(
            3, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1
        )
    
    def _create_strided_conv(self, in_ch, out_ch, stride, kernel_size, padding):
        """Create a strided convolution layer."""
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=1)
    
    def _create_readout_block(self, channels, ro_N, ro_ksize, norm):
        """Create a readout block."""
        padding = ro_ksize // 2
        if self.ro_fcn is "full":
            return nn.Sequential(
                ReadOutConv(
                    inch=channels,
                    outch=channels,
                    ro_N=ro_N,
                    kernel_size=ro_ksize,
                    stride=1,
                    padding=padding,
                ),
                ResBlock(
                    FF(
                        inch=channels,
                        outch=channels,
                        hidch=channels,
                        kernel_size=ro_ksize,
                        stride=1,
                        padding=padding,
                        norm=norm,
                    )
                ),
                BNReLUConv2d(
                    inch=channels,
                    outch=channels,
                    kernel_size=ro_ksize,
                    stride=1,
                    padding=padding,
                    norm=norm,
                )
            )
        elif self.ro_fcn is "norm_id":
             ReadOutConv(
                inch=channels,
                outch=channels,
                ro_N=ro_N,
                kernel_size=ro_ksize,
                stride=1,
                padding=padding,
            )
    
    def _create_layers(self, L, channels, strides, hw_sizes, J, J_bias, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg):
        """Create all network layers."""
        layers = nn.ModuleList()
        padding = ro_ksize // 2
        
        for l in range(L):
            # Create transition layers
            if l == 0:
                transition_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
            else:
                conv_layer = self._create_strided_conv(
                    channels[l-1], channels[l], strides[l], ro_ksize, padding
                )
                transition_layers = nn.ModuleList([conv_layer, conv_layer])
            
            if self.ro_only[l]:
                k_layer = nn.Identity()  # No K-layer for readout-only
            else: 
                # Create K-layer
                k_layer = KLayer(
                    n=self.ns[l],
                    ch=channels[l],
                    J=J[l],
                    J_bias=J_bias,          # Turned off bias in J! by SK, Jul 4 2025
                    c_norm=c_norm,
                    use_omega=use_omega,
                    init_omg=init_omg,
                    global_omg=global_omg,
                    learn_omg=learn_omg,
                    ksize=ksizes[l],
                    hw=hw_sizes[l],
                    bp_steps=self.bp_steps[l],
                )
            
            # Create readout block
            readout_block = self._create_readout_block(channels[l], ro_N[l], ro_ksize, norm)
            
            layers.append(nn.ModuleList([
                transition_layers,
                nn.Identity(),
                k_layer,
                readout_block,
                nn.Identity()
            ]))
        
        return layers
    
    def feature(self, inp):
        """Extract features from input through the network layers."""
        # Initial processing
        c = self.conv0(self.rgb_normalize(inp))
        x = torch.randn_like(c)
        xs, es = [], []

        # Process through each layer
        for l, (transition_layer, _, k_layer, readout_layer, _) in enumerate(self.layers):
            x, c = transition_layer[0](x), transition_layer[1](c)
            if self.ro_only[l] == False:
                layer_xs, layer_es = k_layer(x, c, self.T[l], self.gamma)
                xs.append(layer_xs)
                es.append(layer_es)
                x = layer_xs[-1]
                c = readout_layer(x)
            else:
                # Readout-only layer, no K-layer processing
                xs.append([x])
                es.append([torch.zeros_like(x)])
                # Use pre-created normalization layer to avoid device mismatch
                c = self.ro_c_norm(c)
                c = normalize(c, self.n)
                c = readout_layer(c) # x is a random sampled matrix, thus use c instead
            
        # Final pooling
        x, c = map(self.pool, (x, c))
        return c, x, xs, es

    def forward(self, inp, return_xs=False, return_es=False):
        """Forward pass through the network.
        
        Args:
            inp: Input tensor
            return_xs: Whether to return intermediate states
            return_es: Whether to return energy states
            
        Returns:
            Logits or tuple of (logits, xs, es) depending on flags
        """
        if self.ensemble == 1:
            c, x, xs, es = self.feature(inp)
            logits = self.out(c)
            
            # Build return value based on flags
            result = [logits]
            if return_xs:
                result.append(xs)
            if return_es:
                result.append(es)
            
            return result[0] if len(result) == 1 else result
        else:
            # Ensemble prediction
            assert not return_xs and not return_es, "Ensemble mode doesn't support returning xs/es"
            logits_list = []
            for _ in range(self.ensemble):
                features, _, _, _ = self.feature(inp)
                logits = self.out(features)
                logits_list.append(logits)
            return torch.stack(logits_list).mean(0)


class AKOrNResNet(nn.Module):
    """
    L-layer AKOrN + ResNet
    """

    def __init__(
            self,
            n=2,
            ch=128,
            T=15,
            J="conv",
            gamma=0.01,
            ksizes=3,
            L=1,
            out_classes=10,
            transform_to_theta=False,
            bp_steps=3,
            ro_only=False,
            ):
        
        super().__init__()
        self.kur1 = MyAKOrN(
            n=n,
            ch=ch,
            L=L,
            J=J,
            T=T,
            ksizes=ksizes,
            gamma=gamma,
            use_omega=True,
            init_omg=1.0,
            global_omg=False,
            learn_omg=True,
            out_classes=out_classes,
            bp_steps=bp_steps,
            ro_only=ro_only,
        )
        self.n  = n
        self.ch = ch
        self.transform_to_theta = transform_to_theta
        # J = self._expand_param(J, L)
        
        # For debugging
        self.c, self.x, self.xs, self.es = None, None, None, None
        # self.x = None
        # self.xs = None
        # self.es = Nones
        
        def make_layer(in_ch, out_ch, blocks, stride):
            # ① 先頭ブロック用の downsample を準備
            downsample = None
            if stride != 1 or in_ch != out_ch:
                downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

            # ② ブロックを積む
            layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_ch, out_ch))  # 2 枚目以降 stride=1
            return nn.Sequential(*layers)

        if self.transform_to_theta and self.n == 2:
            dim = ch // n
        else:
            dim = ch

        # To compare to 
        self.layer1 = make_layer(dim, 2*dim, 2, 2)
        self.layer2 = make_layer(2*dim, 4*dim, 2, 2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(4*dim, out_classes)

        # self.layer1 = make_layer(dim, dim, 2, 1)
        # self.layer2 = make_layer(dim, 2*dim, 2, 2)
        # self.layer3 = make_layer(2*dim, 4*dim, 2, 2)
        # self.layer4 = make_layer(4*dim, 8*dim, 2, 2)
        # self.pool   = nn.AdaptiveAvgPool2d(1)
        # self.fc     = nn.Linear(8*dim, out_classes)


    def forward(self, inp):
        _, _, xs, es = self.kur1.feature(inp)
        self.xs = xs
        self.es = es

        self.x = xs[-1][-1]
        self.c = self.kur1.layers[-1][3](self.x)
        
        if self.transform_to_theta and self.n==2:
            y = torch.atan2(self.c[:,1::2,:,:], self.c[:,0::2,:,:])
        else:
            y = self.c
        
        y = self.layer1(y)
        y = self.layer2(y)
        #y = self.layer3(y)
        #y = self.layer4(y)
        y = self.pool(y).flatten(1)
        return self.fc(y)

# class ReadoutOnlyMyAKOrN(MyAKOrN):
#     """
#     Control version of MyAKOrN that uses ControlKLayer instead of KLayer.
#     This removes AKOrN iterations but keeps the readout processing intact.
#     """

#     def __init__(
#         self,
#         n=4,
#         ch=64,
#         out_classes=10,
#         L=3,
#         ro_ksize=3,
#         ro_N=2,
#         norm="bn",
#         c_norm="gn",
#         gamma=1.0,
#         use_omega=True,
#         init_omg=1.0,
#         global_omg=True,
#         learn_omg=True,
#         ensemble=1,
#         bp_steps=None,
#     ):
#         super().__init__()
        
#         self.fixed_noise = False
#         self.n = n
#         self.L = L
#         self.ensemble = ensemble
#         self.gamma = nn.Parameter(torch.tensor([gamma]), requires_grad=False)
        
#         # Expand parameters to match number of layers
#         self.ns = self._expand_param(n, L)
#         self.T = self._expand_param(T, L)
#         self.bp_steps = self._expand_param(bp_steps,L)
#         J = self._expand_param(J, L)
#         ksizes = self._expand_param(ksizes, L)
#         ro_N = self._expand_param(ro_N, L)
                
#         # Calculate layer dimensions
#         strides = [2, 2, 2] + [1] * (L - 3)
#         channels = [ch * (2**i) for i in range(L)]
#         hw_sizes = [[32 // (2**i), 32 // (2**i)] if i < 3 else [8, 8] for i in range(L)]
        
#         # Build network components
#         self.rgb_normalize = RGBNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         self.conv0 = self._create_initial_conv(channels[0])
#         self.layers = self._create_control_layers(L, channels, strides, hw_sizes, J, J_bias, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg)
#         self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Reshape(-1, channels[-1]))
#         self.out = nn.Linear(channels[-1], out_classes)
    
#     def _expand_param(self, param, length):
#         """Expand parameter to match the number of layers."""
#         if isinstance(param, (list, tuple)):
#             return list(param) if len(param) == length else [param[0]] * length
#         return [param] * length
    
#     def _create_initial_conv(self, out_channels):
#         """Create the initial convolutional layer."""
#         return nn.Conv2d(
#             3, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1
#         )
    
#     def _create_strided_conv(self, in_ch, out_ch, stride, kernel_size, padding):
#         """Create a strided convolution layer."""
#         return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=1)
    
#     def _create_readout_block(self, channels, ro_N, ro_ksize, norm):
#         """Create a readout block - KEPT UNCHANGED for fair comparison."""
#         padding = ro_ksize // 2
#         return nn.Sequential(
#             ReadOutConv(
#                 inch=channels,
#                 outch=channels,
#                 ro_N=ro_N,
#                 kernel_size=ro_ksize,
#                 stride=1,
#                 padding=padding,
#             ),
#             ResBlock(
#                 FF(
#                     inch=channels,
#                     outch=channels,
#                     hidch=channels,
#                     kernel_size=ro_ksize,
#                     stride=1,
#                     padding=padding,
#                     norm=norm,
#                 )
#             ),
#             BNReLUConv2d(
#                 inch=channels,
#                 outch=channels,
#                 kernel_size=ro_ksize,
#                 stride=1,
#                 padding=padding,
#                 norm=norm,
#             )
#         )
    
#     def _create_control_layers(self, L, channels, strides, hw_sizes, J, J_bias, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg):
#         """Create all network layers using ControlKLayer instead of KLayer."""
#         layers = nn.ModuleList()
#         padding = ro_ksize // 2
        
#         for l in range(L):
#             # Create transition layers
#             if l == 0:
#                 transition_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
#             else:
#                 conv_layer = self._create_strided_conv(
#                     channels[l-1], channels[l], strides[l], ro_ksize, padding
#                 )
#                 transition_layers = nn.ModuleList([conv_layer, conv_layer])
            
#             # Create Control K-layer (NO AKORN ITERATIONS)
#             control_k_layer = ControlKLayer(
#                 n=self.ns[l],
#                 ch=channels[l],
#                 J=J[l],
#                 J_bias=J_bias,
#                 c_norm=c_norm,
#                 use_omega=use_omega,
#                 init_omg=init_omg,
#                 global_omg=global_omg,
#                 learn_omg=learn_omg,
#                 ksize=ksizes[l],
#                 hw=hw_sizes[l],
#                 bp_steps=self.bp_steps[l],
#             )
            
#             # Create readout block - KEPT UNCHANGED
#             readout_block = self._create_readout_block(channels[l], ro_N[l], ro_ksize, norm)
            
#             layers.append(nn.ModuleList([
#                 transition_layers,
#                 nn.Identity(),
#                 control_k_layer,  # Using ControlKLayer instead of KLayer
#                 readout_block,
#                 nn.Identity()
#             ]))
        
#         return layers
    
#     def feature(self, inp):
#         """Extract features from input through the network layers."""
#         # Initial processing
#         c = self.conv0(self.rgb_normalize(inp))
#         x = torch.randn_like(c)
#         xs, es = [], []

#         # Process through each layer (same structure as original)
#         for l, (transition_layer, _, control_k_layer, readout_layer, _) in enumerate(self.layers):
#             x, c = transition_layer[0](x), transition_layer[1](c)
#             layer_xs, layer_es = control_k_layer(x, c, self.T[l], self.gamma)  # ControlKLayer call
#             xs.append(layer_xs)
#             es.append(layer_es)
#             x = layer_xs[-1]
#             c = readout_layer(x)  # AKORN READOUT PRESERVED
            
#         # Final pooling
#         x, c = map(self.pool, (x, c))
#         return c, x, xs, es

#     def forward(self, inp, return_xs=False, return_es=False):
#         """Forward pass through the network."""
#         if self.ensemble == 1:
#             c, x, xs, es = self.feature(inp)
#             logits = self.out(c)
            
#             # Build return value based on flags
#             result = [logits]
#             if return_xs:
#                 result.append(xs)
#             if return_es:
#                 result.append(es)
            
#             return result[0] if len(result) == 1 else result
#         else:
#             # Ensemble prediction
#             assert not return_xs and not return_es, "Ensemble mode doesn't support returning xs/es"
#             logits_list = []
#             for _ in range(self.ensemble):
#                 features, _, _, _ = self.feature(inp)
#                 logits = self.out(features)
#                 logits_list.append(logits)
#             return torch.stack(logits_list).mean(0)



class ReadoutOnlyMyAKOrN(MyAKOrN):
    """
    Alternative control version of MyAKOrN that inherits from the original MyAKOrN class.
    This approach overrides only the layer creation method to use ControlKLayer instead of KLayer,
    making the code more maintainable and ensuring compatibility with the base class.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize the parent class normally
        super().__init__(*args, **kwargs)
        
    def _create_layers(self, L, channels, strides, hw_sizes, J, J_bias, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg):
        """
        Override the layer creation method to use ControlKLayer instead of KLayer.
        This is the only method that needs to be changed to create the control version.
        """
        layers = nn.ModuleList()
        padding = ro_ksize // 2
        
        for l in range(L):
            # Create transition layers (same as parent)
            if l == 0:
                transition_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
            else:
                conv_layer = self._create_strided_conv(
                    channels[l-1], channels[l], strides[l], ro_ksize, padding
                )
                transition_layers = nn.ModuleList([conv_layer, conv_layer])
            
            # Create Control K-layer (NO AKORN ITERATIONS) - this is the key difference
            control_k_layer = ReadoutOnlyKLayer(
                n=self.ns[l],
                ch=channels[l],
                c_norm=c_norm,
            )
                        
            # Create readout block (same as parent - PRESERVED)
            readout_block = self._create_readout_block(channels[l], ro_N[l], ro_ksize, norm)
            
            layers.append(nn.ModuleList([
                transition_layers,
                nn.Identity(),
                control_k_layer,  # Using ControlKLayer instead of KLayer
                readout_block,
                nn.Identity()
            ]))
        
        return layers




class ROOnlyAKOrNResNet(nn.Module):
    """
    L-layer readout-only AKOrN + ResNet
    """

    def __init__(
            self,
            n=2,
            ch=128,
            L=1,
            out_classes=10,
            transform_to_theta=False,
            ):
        
        super().__init__()
        self.kur1 = ReadoutOnlyMyAKOrN(
            n=n,
            ch=ch,
            L=L,
            out_classes=out_classes,
        )
        self.n  = n
        self.ch = ch
        self.transform_to_theta = transform_to_theta
        # J = self._expand_param(J, L)
        
        # For debugging
        self.c, self.x, self.xs, self.es = None, None, None, None
        
        def make_layer(in_ch, out_ch, blocks, stride):
            # ① 先頭ブロック用の downsample を準備
            downsample = None
            if stride != 1 or in_ch != out_ch:
                downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

            # ② ブロックを積む
            layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_ch, out_ch))  # 2 枚目以降 stride=1
            return nn.Sequential(*layers)

        if self.transform_to_theta and self.n == 2:
            dim = ch // n
        else:
            dim = ch

        # To compare to 
        self.layer1 = make_layer(dim, 2*dim, 2, 2)
        self.layer2 = make_layer(2*dim, 4*dim, 2, 2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(4*dim, out_classes)

        # self.layer1 = make_layer(dim, dim, 2, 1)
        # self.layer2 = make_layer(dim, 2*dim, 2, 2)
        # self.layer3 = make_layer(2*dim, 4*dim, 2, 2)
        # self.layer4 = make_layer(4*dim, 8*dim, 2, 2)
        # self.pool   = nn.AdaptiveAvgPool2d(1)
        # self.fc     = nn.Linear(8*dim, out_classes)


    def forward(self, inp):
        _, _, xs, es = self.kur1.feature(inp)
        self.xs = xs
        self.es = es

        self.x = xs[-1][-1]
        self.c = self.kur1.layers[-1][3](self.x)
        
        if self.transform_to_theta and self.n==2:
            y = torch.atan2(self.c[:,1::2,:,:], self.c[:,0::2,:,:])
        else:
            y = self.c
        
        y = self.layer1(y)
        y = self.layer2(y)
        #y = self.layer3(y)
        #y = self.layer4(y)
        y = self.pool(y).flatten(1)
        return self.fc(y)

class MyAKOrNSimplerReadout(nn.Module):
    """
    Artficial Kuramoto Oscillator Neurons (AKOrN) for classification tasks.
    
    Args:
        n: Number of nodes per layer (int or list)
        ch: Base number of channels
        out_classes: Number of output classes
        L: Number of layers
        T: Number of time steps (int or list)
        J: Connectivity type (str or list)
        J_bias: convolution bias (t or f), added by SK on Jul 4, 2025       
        ksizes: Kernel sizes (int or list)
        ro_ksize: Readout kernel size
        ro_N: Readout N parameter (int or list)
        norm: Normalization type
        c_norm: normalization type for condtional stimuli (c in the paper)
        gamma: Gamma parameter
        use_omega: Whether to use omega
        init_omg: Initial omega value
        global_omg: Whether to use global omega
        learn_omg: Whether to learn omega
        ensemble: Ensemble size
    """

    def __init__(
        self,
        n=2,
        ch=64,
        out_classes=10,
        L=3,
        T=3,
        J="conv",
        J_bias=False,
        ksizes=[9, 7, 5],
        ro_ksize=3,
        ro_N=2,
        norm="bn",
        c_norm="gn",
        gamma=1.0,
        use_omega=True,
        init_omg=1.0,
        global_omg=False,
        learn_omg=True,
        ensemble=1,
    ):
        super().__init__()
        
        self.fixed_noise = False
        self.n = n
        self.L = L
        self.ensemble = ensemble
        self.gamma = nn.Parameter(torch.tensor([gamma]), requires_grad=False)
        
        # Expand parameters to match number of layers
        self.ns = self._expand_param(n, L)
        self.T = self._expand_param(T, L)
        J = self._expand_param(J, L)
        ksizes = self._expand_param(ksizes, L)
        ro_N = self._expand_param(ro_N, L)
        
        # Calculate layer dimensions
        strides = [2, 2, 2] + [1] * (L - 3)
        channels = [ch * (2**i) for i in range(L)]
        hw_sizes = [[32 // (2**i), 32 // (2**i)] if i < 3 else [8, 8] for i in range(L)]
        
        # Build network components
        self.rgb_normalize = RGBNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.conv0 = self._create_initial_conv(channels[0])
        self.layers = self._create_layers(L, channels, strides, hw_sizes, J, J_bias, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Reshape(-1, channels[-1]))
        self.out = nn.Linear(channels[-1], out_classes)
    
    def _expand_param(self, param, length):
        """Expand parameter to match the number of layers."""
        if isinstance(param, (list, tuple)):
            return list(param) if len(param) == length else [param[0]] * length
        return [param] * length
    
    def _create_initial_conv(self, out_channels):
        """Create the initial convolutional layer."""
        return nn.Conv2d(
            3, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1
        )
    
    def _create_strided_conv(self, in_ch, out_ch, stride, kernel_size, padding):
        """Create a strided convolution layer."""
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=1)
    
    def _simple_readout_block(self, channels):
        """Very simple readout"""


    def _create_readout_block(self, channels, ro_N, ro_ksize, norm):
        """Create a readout block."""
        padding = ro_ksize // 2
        return nn.Sequential(
            ReadOutConv(
                inch=channels,
                outch=channels,
                ro_N=ro_N,
                kernel_size=ro_ksize,
                stride=1,
                padding=padding,
            ),
            # ResBlock(
            #     FF(
            #         inch=channels,
            #         outch=channels,
            #         hidch=channels,
            #         kernel_size=ro_ksize,
            #         stride=1,
            #         padding=padding,
            #         norm=norm,
            #     )
            # ),
            # BNReLUConv2d(
            #     inch=channels,
            #     outch=channels,
            #     kernel_size=ro_ksize,
            #     stride=1,
            #     padding=padding,
            #     norm=norm,
            # )
        )
    
    def _create_layers(self, L, channels, strides, hw_sizes, J, J_bias, ksizes, ro_N, ro_ksize, norm, c_norm, use_omega, init_omg, global_omg, learn_omg):
        """Create all network layers."""
        layers = nn.ModuleList()
        padding = ro_ksize // 2
        
        for l in range(L):
            # Create transition layers
            if l == 0:
                transition_layers = nn.ModuleList([nn.Identity(), nn.Identity()])
            else:
                conv_layer = self._create_strided_conv(
                    channels[l-1], channels[l], strides[l], ro_ksize, padding
                )
                transition_layers = nn.ModuleList([conv_layer, conv_layer])
            
            # Create K-layer
            k_layer = KLayer(
                n=self.ns[l],
                ch=channels[l],
                J=J[l],
                J_bias=J_bias,          # Turned of bias in J! by SK, Jul 4 2025
                c_norm=c_norm,
                use_omega=use_omega,
                init_omg=init_omg,
                global_omg=global_omg,
                learn_omg=learn_omg,
                ksize=ksizes[l],
                hw=hw_sizes[l],
            )
            
            # Create readout block
            readout_block = self._create_readout_block(channels[l], ro_N[l], ro_ksize, norm)
            
            layers.append(nn.ModuleList([
                transition_layers,
                nn.Identity(),
                k_layer,
                readout_block,
                nn.Identity()
            ]))
        
        return layers

    def feature(self, inp):
        """Extract features from input through the network layers."""
        # Initial processing
        c = self.conv0(self.rgb_normalize(inp))
        x = torch.randn_like(c)
        xs, es = [], []

        # Process through each layer
        for l, (transition_layer, _, k_layer, readout_layer, _) in enumerate(self.layers):
            x, c = transition_layer[0](x), transition_layer[1](c)
            layer_xs, layer_es = k_layer(x, c, self.T[l], self.gamma)
            xs.append(layer_xs)
            es.append(layer_es)
            x = layer_xs[-1]
            c = readout_layer(x)
            
        # Final pooling
        x, c = map(self.pool, (x, c))
        return c, x, xs, es

    def forward(self, inp, return_xs=False, return_es=False):
        """Forward pass through the network.
        
        Args:
            inp: Input tensor
            return_xs: Whether to return intermediate states
            return_es: Whether to return energy states
            
        Returns:
            Logits or tuple of (logits, xs, es) depending on flags
        """
        if self.ensemble == 1:
            c, x, xs, es = self.feature(inp)
            logits = self.out(c)
            
            # Build return value based on flags
            result = [logits]
            if return_xs:
                result.append(xs)
            if return_es:
                result.append(es)
            
            return result[0] if len(result) == 1 else result
        else:
            # Ensemble prediction
            assert not return_xs and not return_es, "Ensemble mode doesn't support returning xs/es"
            logits_list = []
            for _ in range(self.ensemble):
                features, _, _, _ = self.feature(inp)
                logits = self.out(features)
                logits_list.append(logits)
            return torch.stack(logits_list).mean(0)

