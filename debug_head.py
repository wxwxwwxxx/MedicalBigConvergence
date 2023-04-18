import random
import time
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from modules.patch_embedding import ConvTokenizerPS,TrilinearTokenizer
from modules.masker import Masker
from modules.data_pipeline import TextProcesser,ImageProcesser
from modules.bert import  BertConfig, BertForMaskedDM
from modules.losses import Masked_MSE
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import ViTAutoEnc
from modules import augment as custom_transform
from torchvision import transforms
import random
import os
import numpy as np
from monai.networks.nets import SwinUNETR
os.environ["TOKENIZERS_PARALLELISM"]="true"
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
img_tokenizer = TrilinearTokenizer().to(device)
tp = TextProcesser()
ip = ImageProcesser(norm_max=1000, norm_min=-1000)
f_dict = ip.get_file_dict("/dataset/medical-beit/HMCT")
dataset = tp.dataset_load(r"/dataset/medical-beit")
dataset = tp.dataset_preprocess(dataset, tokenizer, ip, f_dict)
m = Masker(0.15,0.5)
t = transforms.Compose([
    # transforms.RandomApply([
        transforms.RandomApply([custom_transform.RandomColorScale3D(0.1)],p=0.5),
        transforms.RandomApply([custom_transform.RandomNoise3D(0.05)],p=0.5),
        transforms.RandomApply([custom_transform.RandomZoom3D(0.2)],p=0.5),
        transforms.RandomApply([custom_transform.RandomShift3D(50)],p=0.5)

    # ], p=0.7),
])
train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=4,
                              collate_fn=lambda x: tp.collect_fn_vl(x, tokenizer, ip, m, t),num_workers=4,persistent_workers=True)
config = BertConfig(text_length=tokenizer.model_max_length, vl_mlp_length=3,max_position_embeddings=1025,output_hidden_states=True)
# model2 = BertForMaskedDM(config).to(device)
model2 = ViTAutoEnc(1,128,16)
optimizer = torch.optim.AdamW(model2.parameters(),betas = (0.9,0.98),eps=1e-6)
scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0)
scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer , start_factor=1e-3, total_iters=500)
loss_recon = Masked_MSE()
# loss_recon = torch.nn.MSELoss()
model2.train()
from matplotlib import pyplot as plt
for epoch in range(100000):
    for i in train_dataloader:

        optimizer.zero_grad()
        input_img = img_tokenizer.downsample(i['masked_img'].to(device).float())
        gt = img_tokenizer.downsample(i['ground_truth'].to(device).float())
        output_vl = model2(input_img,
            # attention_mask=i['attention_mask'][:, 512:].to(device),
            # token_type_ids=i['token_type_ids'][:, 512:].to(device)
        )[0]
        # loss = loss_recon(output_vl["reconstruction"],gt,i["img_loss_mask"].to(device))
        loss = loss_recon(output_vl,gt,i["img_loss_mask"].to(device))
        loss.backward()

        # bert.encoder.layer.11.attention.self.key.weight
        # bert.encoder.layer.11.attention.self.query.weight
        # bert.encoder.layer.11.attention.self.value.weight
        print(torch.abs(model2.get_parameter("blocks.11.attn.qkv.weight").grad).mean())
        print(torch.abs(model2.get_parameter("blocks.11.mlp.linear2.weight").grad).mean())
        # print(torch.abs(model2.get_parameter("bert.encoder.layer.11.attention.self.query.weight").grad).mean())
        # print(torch.abs(model2.get_parameter("bert.encoder.layer.11.attention.self.value.weight").grad).mean())
        exit()

                # print(i['attention_mask'].size())
        # print(i['attention_mask'][0,::4])
        # attention_mask = i['attention_mask'][0][513:].view([8,8,8]).numpy()
        # mask_img = i["img_loss_mask"].to(device).numpy()
        # r = random.randint(0,127)
        #
        #
        # plt.subplot(2,2,1)
        # plt.imshow(input_img[0,0,r,:,:],vmin=0.0, vmax=1.0)
        # plt.subplot(2,2,2)
        # plt.imshow(gt[0,0,r,:,:],vmin=0.0, vmax=1.0)
        # plt.subplot(2,2,3)
        # plt.imshow(mask_img[0,r//16,:,:],vmin=0.0, vmax=1.0)
        # plt.subplot(2,2,4)
        # plt.imshow(attention_mask[r//16,:,:],vmin=0.0, vmax=1.0)
        # plt.title(f"Slice:{r}")
        # plt.show()