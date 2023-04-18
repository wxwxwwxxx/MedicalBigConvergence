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
from scipy import ndimage
from monai.networks.blocks.transformerblock import TransformerBlock

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
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

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
                 upsampler: nn.Module,
                 hidden_size: int = 768,
                 spatial_dims: int = 3,
                 cls: bool = True
                 ):
        super().__init__()
        # TODO:assuming uniform spatical dim
        if type(img_size) == list:
            img_size = img_size[0]
        if type(patch_size) == list:
            patch_size = patch_size[0]
        self.spatial_size = img_size // patch_size
        self.spatial_dims = spatial_dims
        self.cls = cls
        self.upsampler = upsampler
        # upscale = pow(img_size // self.spatial_size, 1 / 2)
        # assert int(upscale) == upscale
        # new_patch_size = [int(upscale)] * self.spatial_dims # auto calaulate upsample ,no debug
        # conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        # # self.conv3d_transpose* is to be compatible with existing 3d model weights.
        # self.conv3d_transpose = conv_trans(hidden_size, deconv_chns, kernel_size=new_patch_size, stride=new_patch_size)
        # self.conv3d_transpose_1 = conv_trans(
        #     in_channels=deconv_chns, out_channels=out_channels, kernel_size=new_patch_size, stride=new_patch_size
        # )
        # self.norm = nn.LayerNorm(hidden_size)

    def forward(self, img_hidden_state):
        # img_hidden_state = self.norm(img_hidden_state)
        if self.cls:
            x = img_hidden_state[:, 1:]
        else:
            x = img_hidden_state
        x = x.transpose(1, 2)
        d = [self.spatial_size for _ in range(self.spatial_dims)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        x = self.upsampler(x)
        # x = self.conv3d_transpose(x)
        # x = self.conv3d_transpose_1(x)
        return x


class PixelShuffle3D(nn.Module):
    def __init__(self):
        super().__init__()

    def pixelshuffle3d(self, img, upscale_factor):
        b, c, h, w, d = img.size()
        ups = upscale_factor
        nc = c / (pow(ups, 3))
        assert nc % 1 == 0
        nc = int(nc)
        img = img.view(b, nc, ups, ups, ups, h, w, d)

        img = img.permute([0, 1, 5, 2, 6, 3, 7, 4])
        img = img.reshape([b, nc, h * ups, w * ups, d * ups])
        return img

    def pixelunshuffle3d(self, img, downscale_factor):
        b, c, h, w, d = img.size()
        ds = downscale_factor
        assert h % ds == 0 and w % ds == 0 and d % ds == 0
        nc = c * pow(ds, 3)
        img = img.view(b, c, h // ds, ds, w // ds, ds, d // ds, ds)
        #              0  1  2        3   4        5   6        7
        img = img.permute([0, 1, 3, 5, 7, 2, 4, 6])
        img = img.reshape(b, nc, h // ds, w // ds, d // ds)
        return img


class TrilinearUpsampler(nn.Module):
    def __init__(self, input_channel=768, output_channel=1):
        super().__init__()
        # for now, the upsample scale is fixed.
        self.upsample_ops1 = torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up_conv1 = torch.nn.Conv3d(input_channel, 256, 3, padding='same')
        self.relu1 = torch.nn.ReLU(inplace=True)

        self.upsample_ops2 = torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up_conv2 = torch.nn.Conv3d(256, 16, 3, padding='same')
        self.relu2 = torch.nn.ReLU(inplace=True)

        self.upsample_ops3 = torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up_conv3 = torch.nn.Conv3d(16, output_channel, 3, padding='same')

    def forward(self, x):
        x = self.upsample_ops1(x)
        x = self.up_conv1(x)
        x = self.relu1(x)

        x = self.upsample_ops2(x)
        x = self.up_conv2(x)
        x = self.relu2(x)

        x = self.upsample_ops3(x)
        x = self.up_conv3(x)
        return x


class ConvTransUpsampler(nn.Module):
    def __init__(self, input_channel=768, output_channel=1):
        super().__init__()
        # for now, the upsample scale is fixed.

        conv_trans = Conv[Conv.CONVTRANS, 3]
        # self.conv3d_transpose* is to be compatible with existing 3d model weights.
        self.conv3d_transpose1 = conv_trans(input_channel, 256, kernel_size=4, stride=4)
        self.conv3d_transpose2 = conv_trans(in_channels=256, out_channels=16, kernel_size=4, stride=4)
        self.conv3d_transpose3 = conv_trans(in_channels=16, out_channels=output_channel, kernel_size=4, stride=4)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d_transpose1(x)
        x = self.relu1(x)

        x = self.conv3d_transpose2(x)
        x = self.relu2(x)

        x = self.conv3d_transpose3(x)
        return x


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, input_channel=768, output_channel=1):
        super().__init__()
        # for now, the upsample scale is fixed.
        # optional input_channel=>768 1024 1280=>12 16 20
        self.pixel_shuffle_ops = PixelShuffle3D()
        self.up_conv1 = torch.nn.Conv3d(input_channel // 64, 512, 3, padding='same')
        self.up_conv2 = torch.nn.Conv3d(8, 192, 3, padding='same')
        self.up_conv3 = torch.nn.Conv3d(3, output_channel, 1, padding='same')
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pixel_shuffle_ops.pixelshuffle3d(x, 4)
        x = self.up_conv1(x)
        x = self.relu1(x)

        x = self.pixel_shuffle_ops.pixelshuffle3d(x, 4)
        x = self.up_conv2(x)
        x = self.relu2(x)

        x = self.pixel_shuffle_ops.pixelshuffle3d(x, 4)
        x = self.up_conv3(x)
        return x


class PixelShuffleUpsampler2D(nn.Module):
    def __init__(self, input_channel=768, output_channel=1, patch_size=16):
        super().__init__()
        available_size = [12, 14, 16, 32]
        if patch_size not in available_size:
            raise NotImplemented("Patch size must be 16 or 32")
        if patch_size == 16:
            self.upsample_ops = torch.nn.Sequential(
                torch.nn.PixelShuffle(4),
                torch.nn.Conv2d(input_channel // 16, 256, 3, 1, padding='same', padding_mode='replicate'),
                torch.nn.ReLU(True),
                torch.nn.PixelShuffle(2),
                torch.nn.Conv2d(64, 36, 3, 1, padding='same', padding_mode='replicate'),
                torch.nn.ReLU(True),
                torch.nn.PixelShuffle(2),
                torch.nn.Conv2d(9, output_channel, 1, 1, padding='same', padding_mode='replicate'),
            )

        elif patch_size == 32:
            self.upsample_ops = torch.nn.Sequential(
                torch.nn.PixelShuffle(4),
                torch.nn.Conv2d(input_channel // 16, 256, 3, 1, padding='same', padding_mode='replicate'),
                torch.nn.ReLU(True),
                torch.nn.PixelShuffle(4),
                torch.nn.Conv2d(16, 36, 3, 1, padding='same', padding_mode='replicate'),
                torch.nn.ReLU(True),
                torch.nn.PixelShuffle(2),
                torch.nn.Conv2d(9, output_channel, 1, 1, padding='same', padding_mode='replicate'),
            )
        elif patch_size == 14:
            self.upsample_ops = torch.nn.Sequential(
                torch.nn.PixelShuffle(2),
                torch.nn.Conv2d(input_channel // 4, 392, 3, 1, padding='same', padding_mode='replicate'),
                torch.nn.ReLU(True),
                torch.nn.PixelShuffle(7),
                torch.nn.Conv2d(8, 16, 3, 1, padding='same', padding_mode='replicate'),
                torch.nn.ReLU(True),
                # torch.nn.PixelShuffle(2),
                torch.nn.Conv2d(16, output_channel, 1, 1, padding='same', padding_mode='replicate'),
            )
        elif patch_size == 12:
            self.upsample_ops = torch.nn.Sequential(
                torch.nn.PixelShuffle(2),
                torch.nn.Conv2d(input_channel // 4, 288, 3, 1, padding='same', padding_mode='replicate'),
                torch.nn.ReLU(True),
                torch.nn.PixelShuffle(3),
                torch.nn.Conv2d(32, 36, 3, 1, padding='same', padding_mode='replicate'),
                torch.nn.ReLU(True),
                torch.nn.PixelShuffle(2),
                torch.nn.Conv2d(9, output_channel, 1, 1, padding='same', padding_mode='replicate'),
            )

    def forward(self, x):
        return self.upsample_ops(x)


class PixelShuffleUpsampler2D336(nn.Module):
    def __init__(self, input_channel=768, output_channel=1):
        # patch_size: 16 or 32
        super().__init__()

    def forward(self, x):
        return self.upsample_ops(x)
