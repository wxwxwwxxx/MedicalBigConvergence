# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from monai.networks.layers import Conv, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}


class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4, pos_embed="conv")

    """

    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            cls: bool = True
            # TODO: all arguments can be added into BertConfig class, including init trunc std. todo later
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")


        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        self.patch_embeddings: nn.Module
        self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
        )
        if cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-2.0, b=2.0)

        self.dropout = nn.Dropout(dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        # embeddings = x
        embeddings = self.dropout(x)
        return embeddings

class PatchEmbedRecover(nn.Module):
    def __init__(self,
                 img_size: Union[Sequence[int], int],
                 patch_size: Union[Sequence[int], int],
                 out_channels: int = 1,
                 deconv_chns: int = 16,
                 hidden_size: int = 768,
                 spatial_dims: int = 3,
                 cls: bool = True
                 ):
        super().__init__()
        #TODO:assuming uniform spatical dim
        if type(img_size) == list:
            img_size = img_size[0]
        if type(patch_size) == list:
            patch_size = patch_size[0]
        self.spatial_size = img_size//patch_size
        self.spatial_dims = spatial_dims
        self.cls = cls
        new_patch_size = [4] * self.spatial_dims
        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        # self.conv3d_transpose* is to be compatible with existing 3d model weights.
        self.conv3d_transpose = conv_trans(hidden_size, deconv_chns, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_transpose_1 = conv_trans(
            in_channels=deconv_chns, out_channels=out_channels, kernel_size=new_patch_size, stride=new_patch_size
        )
    def forward(self,img_hidden_state):
        if self.cls:
            x = img_hidden_state[:,1:]
        else:
            x = img_hidden_state
        x = x.transpose(1, 2)
        d = [self.spatial_size  for _ in range(3)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)
        return x


class ConvTokenizerPS(nn.Module):
    def __init__(self):
        super().__init__()
        #input:fixed 512*512*512
        self.conv_downsample = nn.Conv3d(64,1,3,1,1)
        self.conv_upsample = nn.Conv3d(1,64,3,1,1)
    def pixelshuffle3d(self,img,upscale_factor):
        b, c, h, w, d = img.size()
        ups = upscale_factor
        nc = c / (pow(ups, 3))
        assert nc % 1 == 0
        nc = int(nc)
        img = img.view(b, nc, ups, ups, ups, h, w, d)

        img = img.permute([0, 1, 5, 2, 6, 3, 7, 4])
        img = img.reshape([b, nc, h * ups, w * ups, d * ups])
        return img
    def pixelunshuffle3d(self,img,upscale_factor):
        b, c, h, w, d = img.size()
        ups = upscale_factor
        assert h % ups == 0 and w % ups == 0 and d % ups == 0
        nc = c * pow(ups, 3)
        img = img.view(b, c, h // ups, ups, w // ups, ups, d // ups, ups)
        #              0 1 2      3   4      5   6      7
        img = img.permute([0, 1, 3, 5, 7, 2, 4, 6])
        img = img.reshape(b, nc, h // ups, w // ups, d // ups)
        return img
    def downsample(self,img):
        img = self.pixelunshuffle3d(img,4)
        img = self.conv_downsample(img)
        return img
    def upsample(self,img):
        img = self.conv_upsample(img)
        img = self.pixelshuffle3d(img, 4)
        return img
    def forward(self,img):
        #for debug only
        ds = self.downsample(img)
        us = self.upsample(ds)
        return us
class ConvTokenizerConvTrans(nn.Module):
    def __init__(self):
        super().__init__()
        #input:fixed 512*512*512
        self.conv_downsample1 = nn.Conv3d(1,16,3,2,1)
        self.conv_downsample2 = nn.Conv3d(16,1,3,2,1)
        self.conv_upsample1 = nn.ConvTranspose3d(1,16,3,2,1,output_padding=1)
        self.conv_upsample2 = nn.ConvTranspose3d(16,1,3,2,1,output_padding=1)
    def downsample(self,img):
        img = self.conv_downsample1(img)
        img = self.conv_downsample2(img)
        return img
    def upsample(self,img):
        img = self.conv_upsample1(img)
        img = self.conv_upsample2(img)
        return img
    def forward(self,img):
        #for debug only
        ds = self.downsample(img)
        us = self.upsample(ds)
        return us


