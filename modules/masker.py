import numpy as np
import torch
import matplotlib.pyplot as plt
import collections
from transformers import default_data_collator

class Masker():
    def __init__(self):
        pass

    def mask_ori_image(self, img, patch_size, mask_prob_img):
        # assuming input is a isotropic cube
        c, h, w, d = img.shape
        assert h == w and w == d, "assuming input is a isotropic cube"
        assert h % patch_size == 0, "length should be able to divided bt patch size"
        n_l = h // patch_size
        img = np.reshape(img, [c, n_l, patch_size, n_l, patch_size, n_l, patch_size])
        img = np.transpose(img, [0, 1, 3, 5, 2, 4, 6])
        img = np.reshape(img, [c, -1, patch_size, patch_size, patch_size])
        mask = np.random.binomial(1, mask_prob_img, img.shape[1])[None, :, None, None, None]
        img *= mask
        img = np.reshape(img, [c, n_l, n_l, n_l, patch_size, patch_size, patch_size])
        img = np.transpose(img, [0, 1, 4, 2, 5, 3, 6])
        img = np.reshape(img, [c, h, w, d])
        return img, mask
    def mask_text(self, features,mask_prob_text,tokenizer):
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
            mask = np.random.binomial(1, mask_prob_text, (len(mapping),))
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

if __name__ == "__main__":
    m = Masker()
    input = np.ones([1, 128,128,128])
    output = m.mask_ori_image(input, 16,0.85)

    from transformers import  AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    text1 = "I'm doing something special. I'm a specialist."
    inputs1 = tokenizer(text1, padding="max_length", truncation=True)
    inputs1["word_ids"] = inputs1.word_ids().copy()
    inputs1["labels"] = inputs1["input_ids"].copy()
    print(inputs1)

    text1 = "测试测试测试测试盲肠炎2",
    inputs2 = tokenizer(text1, padding="max_length", truncation=True)
    inputs2["word_ids"] = inputs2.word_ids().copy()
    inputs2["labels"] = inputs2["input_ids"].copy()
    print(inputs2)

    batch = m.mask_text([inputs1, inputs2],0.5,tokenizer)
    #
    # from modules.bert import BertConfig, BertForMaskedLM, BertForMaskedDM
    # config = BertConfig(text_length=tokenizer.model_max_length, vl_mlp_length=3)
    # model2 = BertForMaskedDM(config)
    # output_text = model2(input_ids=batch['input_ids'], labels=batch['input_ids'])
    # for k in output_text:
    #     print(k, output_text[k].size())