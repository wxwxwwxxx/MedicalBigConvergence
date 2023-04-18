import os
import clip
import torch
from torchvision import transforms
from clip.model import CLIP
import numpy as np
from PIL import Image
print(clip.available_models())
device = torch.device("cuda:3")
os.environ["http_proxy"] = "http://172.17.146.34:8891"
os.environ["https_proxy"] = "http://172.17.146.34:8891"
model, preprocess = clip.load('ViT-L/14@336px', device)
img = torch.rand([4,1,256,256],requires_grad=False).to(device)
def stack2_3channel(x):
    x = torch.cat([x for _ in range(3)],1)
    print(x.size())
    return x
clip_transform = transforms.Compose(
    [
        stack2_3channel,
        transforms.Resize(336),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
)

img = clip_transform(img)
print(img)
f = model.encode_image(img.detach())[:,1:,:]
print(preprocess)
print(f)
