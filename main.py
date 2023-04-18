from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from modules.patch_embedding import ConvTokenizerPS, TrilinearTokenizer
from modules.masker import Masker
from modules.data_pipeline import TextProcesser, ImageProcesser
from modules.bert import BertConfig, BertForMaskedDM
from modules.losses import Masked_MSE
from torch.utils.tensorboard import SummaryWriter

from modules import augment as custom_transform
from torchvision import transforms
import random
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
writer = SummaryWriter('/ckpt/medical-beit/debug')
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
img_tokenizer = TrilinearTokenizer().to(device)
tp = TextProcesser()
ip = ImageProcesser(norm_max=1000, norm_min=-1000)
f_dict = ip.get_file_dict("/dataset/medical-beit/HMCT")
dataset = tp.dataset_load(r"/dataset/medical-beit")
dataset = tp.dataset_preprocess(dataset, tokenizer, ip, f_dict)
m = Masker(0.15, 0.5)
t = transforms.Compose([
    # transforms.RandomApply([
    transforms.RandomApply([custom_transform.RandomColorScale3D(0.1)], p=0.5),
    transforms.RandomApply([custom_transform.RandomNoise3D(0.05)], p=0.5),
    transforms.RandomApply([custom_transform.RandomZoom3D(0.2)], p=0.5),
    transforms.RandomApply([custom_transform.RandomShift3D(50)], p=0.5)

    # ], p=0.7),
])
train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=4,
                              collate_fn=lambda x: tp.collect_fn_vl(x, tokenizer, ip, m, t), num_workers=6,
                              persistent_workers=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

config = BertConfig(text_length=tokenizer.model_max_length, vl_mlp_length=3, max_position_embeddings=1025,output_attentions=True)
model2 = BertForMaskedDM(config).to(device)
optimizer = torch.optim.AdamW(model2.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.05)
scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0)
scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1, total_iters=500)
loss_recon = Masked_MSE()

model2.train()
g_step = 0
for epoch in range(100000):
    for i in train_dataloader:
        optimizer.zero_grad()
        input_img = img_tokenizer.downsample(i['masked_img'].to(device).float())
        gt = img_tokenizer.downsample(i['ground_truth'].to(device).float())
        output_vl = model2(
            image_pixels=input_img,
            input_ids=i['input_ids'].to(device),
            labels=i['labels'].to(device),
            attention_mask=i['attention_mask'].to(device),
            token_type_ids=i['token_type_ids'].to(device)
        )
        loss_image = loss_recon(output_vl["reconstruction"], gt, i["img_loss_mask"].to(device))

        for j in output_vl["attentions"]:
            print(j.size())
        loss = loss_image + output_vl['loss_text'] * 0.01
        loss.backward()
        # for n,i in model2.named_parameters():
        #     print(n,i.grad.size() if i.grad is not None else None,torch.abs(i.grad).mean() if i.grad is not None else None)
        # exit()
        optimizer.step()
        if g_step <= 500:
            scheduler_linear.step()
        else:
            scheduler_cos.step()
        print(
            f"epoch={epoch},g_step={g_step},lr = {optimizer.param_groups[-1]['lr']},loss_image={loss_image.item()},loss_text={output_vl['loss_text']}")
        writer.add_scalar("loss_image", loss_image, global_step=g_step)
        writer.add_scalar("loss_text", output_vl["loss_text"], global_step=g_step)
        writer.add_scalar("lr", optimizer.param_groups[-1]['lr'], global_step=g_step)
        if g_step % 10 == 0:
            writer.add_images("img_gt", gt[:, :, 32, :, :], global_step=g_step)
            writer.add_images("recon", output_vl["reconstruction"][:, :, 32, :, :], global_step=g_step)
            writer.add_images("input_img", input_img[:, :, 32, :, :], global_step=g_step)
        g_step += 1

    # def forward(
    #         self,
    #         image_pixels: Optional[torch.Tensor] = None,
    #         image_embeds: Optional[torch.Tensor] = None,
    #         input_ids: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         token_type_ids: Optional[torch.Tensor] = None,
    #         position_ids: Optional[torch.Tensor] = None,
    #         head_mask: Optional[torch.Tensor] = None,
    #         inputs_embeds: Optional[torch.Tensor] = None,
    #         encoder_hidden_states: Optional[torch.Tensor] = None,
    #         encoder_attention_mask: Optional[torch.Tensor] = None,
    #         labels: Optional[torch.Tensor] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         ground_truth: Optional[torch.Tensor] = None,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[Tuple[torch.Tensor], MaskedDMOutput]:
