import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import collections
from transformers import default_data_collator


class Masker():
    def __init__(self, mask_prob_img, mask_prob_text, img_value_threshold=0.1):
        self.mask_prob_img = mask_prob_img
        self.mask_prob_text = mask_prob_text
        self.img_value_threshold = img_value_threshold
    def mask_ori_image(self, img, patch_size):
        # assuming input is a isotropic cube
        c, h, w, d = img.shape
        assert h == w and w == d, "assuming input is a isotropic cube"
        assert h % patch_size == 0, "length should be able to divided bt patch size"
        assert c == 1, "for now, we only support 1 channel image"
        n_l = h // patch_size
        img = np.reshape(img, [c, n_l, patch_size, n_l, patch_size, n_l, patch_size])
        img = np.transpose(img, [0, 1, 3, 5, 2, 4, 6])
        img = np.reshape(img, [c, -1, patch_size, patch_size, patch_size])
        mask = []
        for i in range(img.shape[1]):
            if np.mean(img[0,i,:,:,:])<self.img_value_threshold:
                mask.append(1)
            else:
                if random.random()<self.mask_prob_img:
                    mask.append(0)
                else:
                    mask.append(1)
        mask = np.array(mask)[None, :, None, None, None]
        img *= mask
        mask = np.where(mask == 0, 1, 0)  # TO FALSE
        img = np.reshape(img, [c, n_l, n_l, n_l, patch_size, patch_size, patch_size])
        img = np.transpose(img, [0, 1, 4, 2, 5, 3, 6])
        img = np.reshape(img, [c, h, w, d])
        mask = np.reshape(mask, [n_l, n_l, n_l])
        return img, mask

    def whole_word_masking(self, input_ids, word_ids, tokenizer):
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
        mask = np.random.binomial(1, self.mask_prob_text, (len(mapping),))
        attn_mask = [1] * len(input_ids)
        labels = input_ids.copy()
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        return input_ids, new_labels, attn_mask


if __name__ == "__main__":
    pass
