import torch
from torch import nn
from monai.networks.nets import ViT,ViTAutoEnc
from modules.patch_embedding import ConvTokenizer
if __name__ == "__main__":
    ti = torch.rand([2,1,512,512,512])
    b = ConvTokenizer()
    ds = b.downsample(ti)
    us = b.upsample(ds)
    print(ds.size())
    print(us.size())