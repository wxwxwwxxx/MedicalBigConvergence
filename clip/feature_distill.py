from .clip import available_models, load
from .model import CLIP

import os
import torch
from torchvision import transforms
from torch import nn
import numpy as np


class CLIPImgFeature(nn.Module):
    def __init__(self, image_size, device):
        super().__init__()

        def stack2_3channel(x):
            x = torch.cat([x for _ in range(3)], 1)
            return x

        self.device = device
        self.clip_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                stack2_3channel,
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.model, _ = load('ViT-L/14@336px', device)
        for i in self.model.parameters():
            i.requires_grad = False
    def get_img_feature(self, x):
        init_device = x.device
        x = x.detach().to(self.device)
        x = self.clip_transform(x)
        feature = self.model.encode_image(x.detach())[:, 1:, :].float().to(init_device)
        return feature
    def forward(self,x):
        return self.get_img_feature(x)

if __name__ == "__main__":
    print(available_models())
    # device = torch.device("cuda:3")
    # os.environ["http_proxy"] = "http://172.17.146.34:8891"
    # os.environ["https_proxy"] = "http://172.17.146.34:8891"
    # model, preprocess = load('ViT-L/14@336px', device)
    # img = torch.rand([4, 1, 256, 256], requires_grad=False).to(device)
    #
    # img = clip_transform(img)
    # print(img)
    # f = model.encode_image(img.detach())[:, 1:, :]
    # print(preprocess)
    # print(f)
