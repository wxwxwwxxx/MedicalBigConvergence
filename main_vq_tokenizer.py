from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from modules.patch_embedding import PatchEmbedRecover, PixelShuffleUpsampler2D, PixelShuffleUpsampler,PixelShuffleUpsampler2D336
from modules.masker import Masker
from modules.data_pipeline import TextProcesser, ImageProcesser
from modules.bert import BertConfig, BertForMaskedDM
from modules.losses import Masked_MSE, Zero_Balance_MSE, Edge_MSE
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import ViTAutoEnc
from modules import augment as custom_transform
from modules.vqvae import VQ_AE, VQ_AE_2D
from torchvision import transforms
import random
import os
from modules.patch_embedding import PatchEmbeddingBlock, PatchEmbedRecover
import numpy as np
from clip import CLIPImgFeature
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
ckpt_root = "/ckpt/medical-vqvae"
expr_tag = "debug12_base_2d_d384_p16_b4_bl64_edgeclip_l1_warmnorm_augr1_clipdis"
# expr_tag = "debug"
device = torch.device("cuda:3")
clip_device = torch.device("cuda:4")
### config
feature_dim_all = 768
warm_up_epoch = 5
codebook_dim = 32
num_heads = 12
ema_beta = 0.99
batch_limit = 64
patch_size = 16
batch_size = 4
img_size = 384

loss_dis = torch.nn.MSELoss()
loss_recon = torch.nn.L1Loss()
# loss_recon = Zero_Balance_MSE(0.001)
loss_edge = Edge_MSE(hw_size=img_size).to(device)

writer = SummaryWriter(os.path.join(ckpt_root, "TBoard", expr_tag))
model_path = os.path.join(os.path.join(ckpt_root, "Models", expr_tag))
os.makedirs(model_path, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tp = TextProcesser()
ip = ImageProcesser(norm_max=1000, norm_min=-1000, cut_incomplete_patch=False)
# f_dict = ip.get_file_dict("/dataset/medical-beit/HMCT")
dataset = ip.dataset_load_img(["/dataset/nii", "/dataset/nii_ori", "/dataset/medical-beit/HMCT"], 0.01)
# dataset = tp.dataset_preprocess(dataset, tokenizer, ip, f_dict)
# m = Masker(0.5, 0.5)
t = transforms.Compose([
    # transforms.RandomApply([
    custom_transform.ZoomtoSize3D(img_size),
    transforms.RandomApply([custom_transform.RandomColorScale3D(0.05)], p=0.5),
    transforms.RandomApply([custom_transform.RandomNoise3D(0.05)], p=0.5),
    transforms.RandomApply([custom_transform.RandomZoom3D(0.2)], p=0.5),
    transforms.RandomApply([custom_transform.RandomShift2D(20)], p=0.5)

    # ], p=0.7),
])
train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=batch_size,
                              collate_fn=lambda x: ip.collect_fn_img_2d(x, t, batch_limit,img_size,patch_size), num_workers=batch_size*2,
                              persistent_workers=True)
# train_dataloader = DataLoader(ds["train"], shuffle=True, batch_size=4,
#                               collate_fn=lambda x: ip.collect_fn_img_2d(x, None))
warm_up_steps = len(train_dataloader) * warm_up_epoch
print(f"warm up stpes:{warm_up_steps}")
vq_ae = VQ_AE_2D(img_size=img_size, patch_size=patch_size, feature_dim=feature_dim_all, mlp_ratio=4.0, num_heads=num_heads,
                 codebook_dim=codebook_dim,
                 output_dim=feature_dim_all, beta=ema_beta).to(device)
ps_upsampler = PixelShuffleUpsampler2D(feature_dim_all, patch_size=patch_size).to(device)

embed_recover = PatchEmbedRecover(img_size, patch_size, upsampler=ps_upsampler, hidden_size=feature_dim_all, cls=False,
                                  spatial_dims=2).to(device)
clip_extractor = CLIPImgFeature(image_size=336,device=clip_device)
optimizer = torch.optim.AdamW([*vq_ae.parameters(), *embed_recover.parameters()], lr=2e-4, betas=(0.9, 0.99),
                              weight_decay=1e-4)
scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * 2, eta_min=1e-5)
# scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1, total_iters=warm_up_steps)

vq_ae.train()
embed_recover.train()
g_step = 0
ind_usage_global = np.array([]).astype("int64")
for epoch in range(100000):
    save_dict = {
        "vq_ae": vq_ae.state_dict(),
        "embed_recover": embed_recover.state_dict(),
        "g_step": g_step,
        "epoch": epoch,
        "optim": optimizer.state_dict()
    }
    torch.save(save_dict, os.path.join(model_path, "latest.pt"))
    for i in train_dataloader:
        warm_up_sign = epoch < warm_up_epoch
        if warm_up_sign:
            optimizer.zero_grad()
            gt = i['ground_truth'].float().to(device)
            clip_feature = clip_extractor(gt)
            output_feature = vq_ae.forward_warmup(gt)
            # print(output_feature.size(),clip_feature.size())
            output_img = embed_recover(output_feature)
            output_vl = {"reconstruction": output_img}

            loss_clip = loss_dis(output_feature,clip_feature)
            loss_img = loss_recon(output_vl["reconstruction"], gt)
            loss_e = loss_edge.forward_clip(output_vl["reconstruction"], gt)
            loss = loss_img + loss_e + loss_clip
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            gt = i['ground_truth'].float().to(device)
            clip_feature = clip_extractor(gt)

            output_feature, l1, l2, indice = vq_ae(gt)
            output_img = embed_recover(output_feature)
            output_vl = {"reconstruction": output_img}

            loss_clip = loss_dis(output_feature, clip_feature)
            loss_img = loss_recon(output_vl["reconstruction"], gt)
            loss_e = loss_edge.forward_clip(output_vl["reconstruction"], gt)
            loss = loss_img + l1 + l2 + loss_e + loss_clip
            loss.backward()
            optimizer.step()
            vq_ae.codebook_update()

        # if g_step <= warm_up_steps:
        #     scheduler_linear.step()
        # else:
        #     scheduler_cos.step()

        scheduler_cos.step()
        print(f"tag={expr_tag},epoch={epoch},g_step={g_step},lr={optimizer.param_groups[-1]['lr']},loss={loss.item()}")
        writer.add_scalar("loss_image", loss_img, global_step=g_step)
        writer.add_scalar("loss_sum", loss, global_step=g_step)
        writer.add_scalar("loss_edge", loss_e, global_step=g_step)
        writer.add_scalar("loss_clip", loss_clip, global_step=g_step)
        writer.add_scalar("lr", optimizer.param_groups[-1]['lr'], global_step=g_step)
        if not warm_up_sign:
            ind_usage = torch.unique(indice.detach().cpu()).numpy()
            ind_usage_global = np.unique(np.concatenate([ind_usage_global, ind_usage]))
            code_usage_batch = len(ind_usage)
            code_usage_global = len(ind_usage_global)
            print(f"code_usage_batch={code_usage_batch},code_usage_global={code_usage_global}")
            writer.add_scalar("code_usage_batch", code_usage_batch, global_step=g_step)
            writer.add_scalar("code_usage_global", code_usage_global, global_step=g_step)
            writer.add_scalar("loss_code", l1, global_step=g_step)
        else:
            print(f"warm_up")
        if g_step % 10 == 0:
            random_index = np.random.randint(0, gt.size()[0], 4)
            writer.add_images("img_gt", gt[random_index, :, :, :], global_step=g_step)
            writer.add_images("recon", output_vl["reconstruction"][random_index, :, :, :], global_step=g_step)
            # writer.add_images("input_img",input_img[:, :, 32, :, :], global_step=g_step)
        g_step += 1
