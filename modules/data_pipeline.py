import SimpleITK as sitk
import numpy as np
import math
from datasets import Dataset
import os
import pandas as pd
import glob
from patch_embedding import ConvTokenizerPS

class ImageProcesser():
    def __init__(self, norm_max, norm_min):
        self.norm_max = norm_max
        self.norm_min = norm_min
        self.medical_img_ext = {".nii", ".mhd"}  # TODO:add some ext afterward

    def padding(self, img, patch_size, target_size, cut_incomplete_patch):
        """
        img size: list of [c,h,w,d]
        patch_size: int
        target_size: int. Should be divisible by patch_size
        cut_incomplete_patch: bool, whether cut patch that not fully filled

        return img, attn_mask
        """
        c, h, w, d = img.shape
        # center_crop

        n_start_h = (h - target_size) // 2 if h > target_size else 0
        n_start_w = (w - target_size) // 2 if w > target_size else 0
        n_start_d = (d - target_size) // 2 if d > target_size else 0

        img = img[:, n_start_h:n_start_h + target_size, n_start_w:n_start_w + target_size,
              n_start_d:n_start_d + target_size]

        c, h, w, d = img.shape

        if h < target_size:
            if cut_incomplete_patch:
                nh = (h // patch_size) * patch_size
                n_start = (h - nh) // 2

                img = img[:, n_start:nh + n_start, :, :]
                h = nh

        if w < target_size:
            if cut_incomplete_patch:
                nw = (w // patch_size) * patch_size
                n_start = (w - nw) // 2
                img = img[:, :, n_start:nw + n_start, :]
                w = nw

        if d < target_size:
            if cut_incomplete_patch:
                nd = (d // patch_size) * patch_size
                n_start = (d - nd) // 2
                img = img[:, :, :, n_start:nd + n_start]
                d = nd

        attn_h = math.ceil(h / patch_size)
        attn_w = math.ceil(w / patch_size)
        attn_d = math.ceil(d / patch_size)
        attn_target = target_size // patch_size
        attn_mask = np.ones([attn_h, attn_w, attn_d],dtype=np.int64)

        img = np.pad(img, [(0, 0), (0, target_size - h), (0, target_size - w), (0, target_size - d)], mode='constant')
        attn_mask = np.pad(attn_mask, [(0, attn_target - attn_h), (0, attn_target - attn_w), (0, attn_target - attn_d)],
                           mode='constant')
        return img, attn_mask

    def get_file_dict(self, file_path):
        f_dict = {}

        for fr, fd, fn in os.walk(file_path):
            for i in fn:
                if os.path.splitext(i)[1] not in self.medical_img_ext:
                    pass
                ac = i.split(".")[1]
                full_path = os.path.join(fr, i)
                size = os.path.getsize(full_path)
                if ac not in f_dict:
                    f_dict[ac] = (full_path, size)
                else:
                    if f_dict[ac][1] < size:
                        f_dict[ac] = (full_path, size)
        return f_dict

    def img_path(self, example, file_path_dict):
        path = None
        if example['检查号'] in file_path_dict:
            path = file_path_dict[example['检查号']][0]
        ret = {"img_path": path}
        return ret

    def load_img(self, path):
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        img = img[None, :, :, :].astype("float32")
        img = (img - self.norm_min) / (self.norm_max - self.norm_min)
        ret, attn_mask = self.padding(img, 64, 512, True)
        return ret, attn_mask


class TextProcesser():
    def __init__(self):
        self.control_char_re = re.compile('\s')

    def dataset_load(self, path):
        f_list = glob.glob(os.path.join(path, "*.xlsx"))
        e_list = []
        for f in f_list:
            e = pd.read_excel(f)
            e_list.append(e)
        e = pd.concat(e_list)
        dataset = Dataset.from_pandas(e)
        dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.3)
        return dataset

    def input_concat(self, example):
        return {"input_text": example['影像所见/病理镜下所见'] + example['影像诊断']}

    def strip_control_characters(self, s):
        temp = self.control_char_re.sub('', s)
        return temp

    def strip_controllers(self, example):
        return {"input_text": self.strip_control_characters(example["input_text"])}

    def tokenize(self, example, tokenizer):
        inputs2 = tokenizer(example["input_text"], truncation=True)
        ret = {"input_ids": inputs2["input_ids"], "word_ids": inputs2.word_ids().copy()}
        return ret

    def dataset_preprocess(self, dataset, tokenizer, ip, f_dict):
        dataset = dataset.map(self.input_concat)
        dataset = dataset.map(self.strip_controllers)
        dataset = dataset.map(lambda x: ip.img_path(x, f_dict))

        rc = [i for i in dataset["train"].column_names if i not in ['img_path', 'input_text']]
        dataset = dataset.map(remove_columns=rc)
        dataset = dataset.filter(lambda x: x["img_path"] is not None)
        dataset = dataset.map(lambda x: self.tokenize(x, tokenizer))
        return dataset

    def list_padding(self, input_list, item, padding_length):
        length = len(input_list)
        p_length = padding_length - length
        input_list += [item] * p_length
        return input_list

    def whole_word_masking(self, input_ids, word_ids, mask_prob_text):
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
        attn_mask = [1] * len(input_ids)
        labels = input_ids.copy()
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        return input_ids, new_labels, attn_mask

    def collect_fn(self, features, mask_prob_text, tokenizer, ip, padding_length=512):
        for feature in features:
            feature.pop("input_text")
            word_ids = feature.pop("word_ids")
            img_path = feature.pop("img_path")
            input_ids = feature.pop("input_ids")
            img, img_attn = ip.load_img(img_path)
            input_ids, new_labels, attn_mask = self.whole_word_masking(input_ids, word_ids, mask_prob_text)
            input_ids = self.list_padding(input_ids, tokenizer.pad_token_id, padding_length)
            new_labels = self.list_padding(new_labels, -100, padding_length)
            attn_mask = self.list_padding(attn_mask, 0, padding_length)
            feature["labels"] = new_labels
            feature["attention_mask"] = attn_mask
            feature["input_ids"] = input_ids
            feature["img"] = img.astype("float16")
            feature["img_attn"] = img_attn
            feature["token_type_ids"] = [0]*padding_length+[1]*512 #TODO: it is fixed in
        return default_data_collator(features)


if __name__ == "__main__":
    #
    #
    #

    import re
    from transformers import AutoTokenizer, default_data_collator
    import collections
    from torch.utils.data import DataLoader
    import torch
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    img_tokenizer = ConvTokenizerPS().half().to(device)
    print(tokenizer.pad_token_id)
    tp = TextProcesser()
    ip = ImageProcesser(norm_max=1000, norm_min=-1000)
    f_dict = ip.get_file_dict("/dataset/medical-beit/HMCT")
    dataset = tp.dataset_load(r"/dataset/medical-beit")
    dataset = tp.dataset_preprocess(dataset, tokenizer, ip, f_dict)

    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=2,
                                  collate_fn=lambda x: tp.collect_fn(x, 0.5, tokenizer, ip))
    for i in train_dataloader:

        input_img = img_tokenizer.downsample(i['img'].to(device))
        print(input_img.size())
