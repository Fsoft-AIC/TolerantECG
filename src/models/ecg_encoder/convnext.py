# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, num_dim, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.num_dim = num_dim
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)

            if self.num_dim == 1: 
                x = self.weight[:, None] * x + self.bias[:, None]
            else:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, num_dim, dim):
        super().__init__()
        self.num_dim = num_dim
        if num_dim == 1:
            self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        if self.num_dim == 1:
            Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        else:
            Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, num_dim, ConvLayer, dim, drop_path=0.):
        super().__init__()
        self.num_dim = num_dim
        self.dwconv = ConvLayer(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(num_dim, dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(num_dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        if self.num_dim == 1:
            x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        else:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.num_dim == 1:
            x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        else:
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """

    def __init__(self, 
                 num_dim=1,
                 in_chans=12,
                 num_classes=768,
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 head_bias=False,
                 last_norm=False,
                 seed=42
                 ):
        if seed is not None:
            torch.manual_seed(seed)

        super().__init__()

        self.num_dim = num_dim
        self.in_chans = in_chans
        
        if num_dim == 1:
            ConvLayer = nn.Conv1d
        else:
            ConvLayer = nn.Conv2d

        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        stem = nn.Sequential(
            ConvLayer(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(num_dim, dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(num_dim, dims[i], eps=1e-6, data_format="channels_first"),
                    ConvLayer(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(num_dim, ConvLayer, dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1]) if last_norm else nn.Identity()
        self.head = nn.Linear(dims[-1], num_classes, head_bias) if num_classes > 0 else nn.Identity()

        self.embed_dim = dims[-1]
        self.out_dim = num_classes if num_classes > 0 else self.embed_dim
        self.head_bias = head_bias
        self.last_norm = last_norm

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        if self.num_dim == 1:
            x = x.mean([-1])    # global average pooling, (N, C, L) -> (N, C)
        else:
            x = x.mean([-2, -1]) # global average pooling, (N, C, H, W) -> (N, C)
        return x

    def forward_head(self, x):
        return self.head(self.norm(x))

    def forward(self, x):
        x = self.forward_features(x)
        return self.forward_head(x)
    
    def forward_intermidiate(self, x):
        res = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            res.append(x)
        if self.num_dim == 1:
            x = x.mean([-1])    # global average pooling, (N, C, L) -> (N, C)
        else:
            x = x.mean([-2, -1]) # global average pooling, (N, C, H, W) -> (N, C)
        res.append(x)

        x = self.norm(x)
        x = self.head(x)
        res.append(x)

        return res
       

if __name__ == "__main__":
    x = torch.randn((4, 12, 5000))
    model = ConvNeXtV2(num_dim=1, in_chans=12, num_classes=2048)

    out = model(x)
    print(out.shape)
    
    # outs = model.forward_inter(x)
    # for out in outs:
    #     print(out.shape)

    # x = torch.randn((4, 12, 5000))
    # model = ConvNeXtV2(num_dim=1)
    # model(x)