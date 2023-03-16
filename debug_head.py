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


#=========
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")
text = "China has announced that [MASK]."
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
import torch

# inputs = tokenizer(text, return_tensors="pt")
# token_logits = model(**inputs).logits
# # Find the location of [MASK] and extract its logits
# mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
# mask_token_logits = token_logits[0, mask_token_index, :]
# # Pick the [MASK] candidates with the highest logits
# top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
#
# for token in top_5_tokens:
#     print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

from datasets import load_dataset

imdb_dataset = load_dataset("imdb")
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
# print(tokenized_datasets["train"]["word_ids"])
chunk_size = 128
# tokenized_samples = tokenized_datasets["train"][:3]
#
# for idx, sample in enumerate(tokenized_samples["input_ids"]):
#     print(f"'>>> Review {idx} length: {len(sample)}'")
# concatenated_examples = {
#     k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
# }
# total_length = len(concatenated_examples["input_ids"])
# print(f"'>>> Concatenated reviews length: {total_length}'")
# chunks = {
#     k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
#     for k, t in concatenated_examples.items()
# }
#
# for chunk in chunks["input_ids"]:
#     print(f"'>>> Chunk length: {len(chunk)}'")
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)
import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)
print(batch)
exit()
# for chunk in batch["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")
train_size = 10_00
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
from transformers import TrainingArguments

batch_size = 64
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=False,
    logging_steps=logging_steps,
    remove_unused_columns=False
)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=whole_word_masking_data_collator,
    tokenizer=tokenizer,
)
import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.train()
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")



