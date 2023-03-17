from transformers import AutoTokenizer
from modules.bert import BertModel, BertConfig, BertForMaskedLM, BertForMaskedDM
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

config = BertConfig(text_length=tokenizer.model_max_length, vl_mlp_length=3)
model2 = BertForMaskedDM(config)
optimizer = torch.optim.Adam(model2.parameters())

text1 = ["测试测试测试测试测试1", "测试测试测试测试测试2"]
inputs = tokenizer(text1, return_tensors='pt', padding="max_length", truncation=True)
inputs["image_pixels"] = torch.rand([2, 1, 128, 128, 128])
inputs = {k: inputs[k] for k in ['input_ids', "image_pixels"]}

print("======")

output_imgs = model2(image_pixels=inputs['image_pixels'], ground_truth=inputs['image_pixels'])
for i in output_imgs:
    print(i, output_imgs[i].size())
output_imgs["loss_image"].backward()
for n, i in model2.named_parameters():
    print(n, i.grad is None or (torch.sum(i.grad) == 0.0).item())
optimizer.step()
optimizer.zero_grad()

print("======")
output_text = model2(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
for k in output_text:
    print(k, output_text[k].size())
output_text["loss_text"].backward()
for n, i in model2.named_parameters():
    print(n, i.grad is None or (torch.sum(i.grad) == 0.0).item())
optimizer.step()
optimizer.zero_grad()
print("======")
output_vl = model2(image_pixels=inputs['image_pixels'], input_ids=inputs['input_ids'],
                   ground_truth=inputs['image_pixels'], labels=inputs['input_ids'])
for k in output_vl:
    print(k, output_vl[k].size())
l = output_vl["loss_image"] + output_vl["loss_text"]
l.backward()
for n, i in model2.named_parameters():
    print(n, i.grad is None or (torch.sum(i.grad) == 0.0).item())
optimizer.step()
optimizer.zero_grad()
print("======")

# for k in output_imgs:
#     print(k,output_imgs[k].size())
# for k in output_vl:
#     print(k,output_vl[k].size())
# for k in output_text:
#     print(k,output_text[k].size())
#
# for n,i in model2.named_parameters():
#     print(n,i.size())
