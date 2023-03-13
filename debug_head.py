from transformers import AutoTokenizer
from modules.bert import BertConfig, BertForMaskedLM, BertForMaskedAE
import torch
tokenizer = AutoTokenizer.from_pretrained("/Users/xinwei/PycharmProjects/pretrain_model/bert-base-hm-wmm-20e-384b-15m")

config = BertConfig(text_length=tokenizer.model_max_length,vl_mlp_length=3)
model2 = BertForMaskedLM(config)
cer = torch.nn.MSELoss()
image = torch.rand([3,1,128,128,128])
gt = torch.rand([3,1,128,128,128])


optimizer = torch.optim.Adam(model2.parameters())
input = {"image_pixels":image,"ground_truth":gt}
ret = model2(**input)
ret.loss.backward()
for n,i in model2.named_parameters():
    print(n,i.size(),i.grad is None)


