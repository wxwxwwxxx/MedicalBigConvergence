from transformers import AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
import torch
from modules.patch_embedding import ConvTokenizerPS
from modules.masker import Masker
from modules.data_pipeline import TextProcesser,ImageProcesser
from modules.bert import BertModel, BertConfig, BertForMaskedLM, BertForMaskedDM


device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
img_tokenizer = ConvTokenizerPS().half().to(device)
tp = TextProcesser()
ip = ImageProcesser(norm_max=1000, norm_min=-1000)
f_dict = ip.get_file_dict("/dataset/medical-beit/HMCT")
dataset = tp.dataset_load(r"/dataset/medical-beit")
dataset = tp.dataset_preprocess(dataset, tokenizer, ip, f_dict)
m = Masker(0.5, 0.5)

train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=2,
                              collate_fn=lambda x: tp.collect_fn_vl(x, tokenizer, ip, m, None))
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

config = BertConfig(text_length=tokenizer.model_max_length, vl_mlp_length=3,max_position_embeddings=1025)
model2 = BertForMaskedDM(config).to(device)
optimizer = torch.optim.Adam(model2.parameters())
loss_recon = torch.nn.MSELoss()
for i in train_dataloader:

    input_img = img_tokenizer.downsample(i['masked_img'].to(device))
    output_vl = model2(
        image_pixels=input_img.float(),
        input_ids=i['input_ids'].to(device),
        labels=i['labels'].to(device),
        attention_mask=i['attention_mask'].to(device),
        token_type_ids=i['token_type_ids'].to(device)
    )
    upsample_recon=img_tokenizer.upsample(output_vl["reconstruction"].half())
    loss_image = loss_recon(upsample_recon,i['ground_truth'].to(device))
    print(loss_image.item())


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