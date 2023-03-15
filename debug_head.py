from transformers import AutoTokenizer
from modules.bert import BertConfig, BertForMaskedLM, BertForMaskedAE
import torch
import os
os.environ["http_proxy"]="http://172.17.146.34:8891"
os.environ["https_proxy"]="http://172.17.146.34:8891"
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

config = BertConfig(text_length=tokenizer.model_max_length,vl_mlp_length=3)
model2 = BertForMaskedAE(config)
cer = torch.nn.MSELoss()
image = torch.rand([3,1,128,128,128])
gt = torch.rand([3,1,128,128,128])


optimizer = torch.optim.Adam(model2.parameters())
input = {"image_pixels":image,"ground_truth":gt}
ret = model2(**input)
print(image.size())
print(ret['reconstruction'])


