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
from modules.patch_embedding import PatchEmbeddingBlock,PatchEmbedRecover
os.environ["TOKENIZERS_PARALLELISM"]="true"
writer = SummaryWriter('/ckpt/medical-beit/debug27_patch_embed_check')
device = torch.device("cuda:2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
img_tokenizer = TrilinearTokenizer().to(device)
tp = TextProcesser()
ip = ImageProcesser(norm_max=1000, norm_min=-1000)
f_dict = ip.get_file_dict("/dataset/medical-beit/HMCT")
dataset = tp.dataset_load(r"/dataset/medical-beit")
dataset = tp.dataset_preprocess(dataset, tokenizer, ip, f_dict)
m = Masker(0.5,0.5)
t = transforms.Compose([
    # transforms.RandomApply([
        transforms.RandomApply([custom_transform.RandomColorScale3D(0.1)],p=0.5),
        transforms.RandomApply([custom_transform.RandomNoise3D(0.05)],p=0.5),
        transforms.RandomApply([custom_transform.RandomZoom3D(0.2)],p=0.5),
        transforms.RandomApply([custom_transform.RandomShift3D(50)],p=0.5)

    # ], p=0.7),
])
train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=4,
                              collate_fn=lambda x: tp.collect_fn_vl(x, tokenizer, ip, m, t),num_workers=6,persistent_workers=True)

config = BertConfig(text_length=tokenizer.model_max_length, vl_mlp_length=3,max_position_embeddings=1025,output_hidden_states=True)
# model2 = BertForMaskedDM(config).to(device)
model1 = PatchEmbeddingBlock(1,128,16,768,12).to(device)
model3 = PatchEmbedRecover(128,16).to(device)
optimizer = torch.optim.AdamW([*model1.parameters(),*model3.parameters()],lr=0.01,betas = (0.9,0.98),eps=1e-6,weight_decay=0.05)
scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0)
scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer , start_factor=1e-5,end_factor=1, total_iters=500)
#loss_recon = Masked_MSE()
loss_recon = torch.nn.MSELoss()

# model2.train()
model1.train()
model3.train()
g_step = 0

for epoch in range(100000):
    for i in train_dataloader:
        optimizer.zero_grad()
        #input_img = img_tokenizer.downsample(i['masked_img'].to(device).float())
        gt = img_tokenizer.downsample(i['ground_truth'].to(device).float())
        output_btnk = model1(gt
            # image_pixels=gt,
            # attention_mask=i['attention_mask'][:,512:].to(device),
            # token_type_ids=i['token_type_ids'][:,512:].to(device)
        )
        output_vl={"reconstruction":model3(output_btnk)}
        loss = loss_recon(output_vl["reconstruction"],gt)
        loss.backward()
        optimizer.step()
        if g_step <= 500:
            scheduler_linear.step()
        else:
            scheduler_cos.step()
        print(f"epoch={epoch},g_step={g_step},lr={optimizer.param_groups[-1]['lr']},loss_image={loss.item()}")
        writer.add_scalar("loss_image", loss, global_step=g_step)
        writer.add_scalar("lr", optimizer.param_groups[-1]['lr'], global_step=g_step)
        # for logi in [1,5,9,12]:
        #     f0 = output_vl["hidden_states"][logi]
        #     c_var = torch.var(torch.mean(f0,1))
        #     p_var = torch.var(torch.mean(f0,2))
        #     p_mask = torch.masked_select(f0[:,1:,:],i['img_loss_mask'].to(device).view(-1,512)[:,:,None] == 1).view(-1,768)
        #     c_var_mask = torch.var(torch.mean(p_mask, 0))
        #     p_var_mask = torch.var(torch.mean(p_mask, 1))
        #     g_d = torch.abs(model2.get_parameter(f"bert.encoder.layer.{logi-1}.output.dense_image.weight").grad).mean()
        #     g_k = torch.abs(model2.get_parameter(f"bert.encoder.layer.{logi-1}.attention.self.key.weight").grad).mean()
        #
        #     writer.add_scalar(f"var_channel{logi-1}", c_var, global_step=g_step)
        #     writer.add_scalar(f"var_patch{logi-1}", p_var, global_step=g_step)
        #     writer.add_scalar(f"var_mask_channel{logi-1}", c_var_mask, global_step=g_step)
        #     writer.add_scalar(f"var_mask_patch{logi-1}", p_var_mask, global_step=g_step)
        #     writer.add_scalar(f"grad_dense{logi-1}", g_d, global_step=g_step)
        #     writer.add_scalar(f"grad_qkv{logi-1}", g_k, global_step=g_step)


        if g_step % 10 == 0:
            writer.add_images("img_gt",gt[:, :, 32, :, :], global_step=g_step)
            writer.add_images("recon", output_vl["reconstruction"][:, :, 32, :, :], global_step=g_step)
            # writer.add_images("input_img",input_img[:, :, 32, :, :], global_step=g_step)
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