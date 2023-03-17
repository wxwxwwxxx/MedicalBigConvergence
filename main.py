import torch
from torch import nn
from monai.networks.nets import ViT,ViTAutoEnc
from modules.patch_embedding import ConvTokenizerConvTrans, ConvTokenizerPS
from torchsummary import summary
import os
if __name__ == "__main__":

    device = torch.device("cuda:0")
    b = ConvTokenizerPS().half().to(device)
    a = torch.rand([2,1,512,512,512]).half().to(device)
    ds = b.downsample(a)
    us = b.upsample(ds)
    print(ds.dtype)
    print(us.size())
    # # print(us == a)
    # a=torch.rand([4,48])
    # b=torch.tensor([0,1,2,3])
    # loss = nn.CrossEntropyLoss()
    # layer = torch.nn.Linear(48,4)
    # layer2 = torch.nn.Linear(4, 4)
    # opti = torch.optim.SGD([*layer.parameters(),*layer2.parameters()],lr=1)
    # # while 1:
    # opti.zero_grad()
    # te = layer(a)
    # la = layer2(te)
    # l = loss(la,b)
    # print(layer.bias.is_leaf)
    # # layer2.bias.backward(gradient=torch.tensor([400.0,400.0,400.0,400.0,]))
    # grad = torch.autograd.grad(outputs=l,inputs=layer.bias, create_graph=True)
    # print(grad           )
    # opti.step()
    #


