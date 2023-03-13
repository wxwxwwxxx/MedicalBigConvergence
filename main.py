import torch
from torch import nn
from monai.networks.nets import ViT,ViTAutoEnc
from modules.patch_embedding import ConvTokenizerConvTrans, ConvTokenizerPS
from torchsummary import summary
if __name__ == "__main__":
    device = torch.device("cuda:0")
    b = ConvTokenizerPS().to(device)
    a = torch.rand([2,1,512,512,512]).to(device)
    ds = b.pixelunshuffle3d(a,4)
    us = b.pixelshuffle3d(ds,4)
    print(ds.size())
    print(us.size())
    print(us == a)



