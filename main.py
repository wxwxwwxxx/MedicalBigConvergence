import torch
from torch import nn
from monai.networks.nets import ViT,ViTAutoEnc
from modules.patch_embedding import ConvTokenizerConvTrans
if __name__ == "__main__":
    b = ConvTokenizerConvTrans()
    a = torch.rand([2,1,512,512,512])
    ds = b.downsample(a)
    us = b.upsample(ds)
    print(ds.size())
    print(us.size())



