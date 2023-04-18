import SimpleITK as sitk
import numpy as np
import math
from datasets import Dataset
import os
import pandas as pd
import glob
import re
import torch
from transformers import default_data_collator
import random

class ImageProcesser():
    def __init__(self, norm_max, norm_min, cut_incomplete_patch):
        self.norm_max = norm_max
        self.norm_min = norm_min
        self.cut_incomplete_patch = cut_incomplete_patch
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
        attn_mask = np.ones([attn_h, attn_w, attn_d], dtype=np.int64)

        img = np.pad(img, [(0, 0), (0, target_size - h), (0, target_size - w), (0, target_size - d)], mode='constant')
        attn_mask = np.pad(attn_mask, [(0, attn_target - attn_h), (0, attn_target - attn_w), (0, attn_target - attn_d)],
                           mode='constant')
        return img, attn_mask

    def padding_xy(self, img, patch_size, target_size, cut_incomplete_patch):
        """
        img size: list of [c,h,w,d]
        patch_size: int
        target_size: int. Should be divisible by patch_size
        cut_incomplete_patch: bool, whether cut patch that not fully filled

        return img, attn_mask
        """
        c, h, w, d = img.shape
        # center_crop

        # n_start_h = (h - target_size) // 2 if h > target_size else 0
        n_start_w = (w - target_size) // 2 if w > target_size else 0
        n_start_d = (d - target_size) // 2 if d > target_size else 0

        img = img[:, :, n_start_w:n_start_w + target_size,
              n_start_d:n_start_d + target_size]

        c, h, w, d = img.shape


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




        img = np.pad(img, [(0, 0), (0, 0), (0, target_size - w), (0, target_size - d)], mode='constant')
        return img

    def get_file_dict(self, file_path, f_dict = None):
        if f_dict is None:
            f_dict = {}
        for fr, fd, fn in os.walk(file_path):
            for i in fn:
                if os.path.splitext(i)[1] not in self.medical_img_ext:
                    continue
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

    def load_img(self, path, img_aug,patch_size,img_size,padding_z=True):
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)

        img = img[None, :, :, :].astype("float32")
        img = (img - self.norm_min) / (self.norm_max - self.norm_min)
        if img_aug is not None:
            img = img_aug(img)
        img = np.clip(img, 0.0, 1.0)
        if padding_z:
            ret, attn_mask = self.padding(img, patch_size, img_size, self.cut_incomplete_patch)
            return ret, attn_mask
        else:
            ret = self.padding_xy(img,patch_size,img_size, self.cut_incomplete_patch)
            return ret, None
    def dataset_load_img(self,path_list,test_ratio):
        fp = None
        for p in path_list:
            fp = self.get_file_dict(p,fp)
        pd_dict = {}
        pd_dict['检查号'] = [k for k in fp.keys()]
        pd_dict['img_path'] = [v[0] for v in fp.values()]
        pd_df = pd.DataFrame(pd_dict)
        dataset = Dataset.from_pandas(pd_df)
        dataset = dataset.shuffle(seed=42).train_test_split(test_size=test_ratio)
        return dataset
    def collect_fn_img(self, features, img_aug,img_size,patch_size):
        for feature in features:
            feature.pop('检查号')
            img_path = feature.pop("img_path")
            img, img_attn = self.load_img(img_path, img_aug,patch_size,img_size)
            feature["ground_truth"] = img.astype("float16")
        return default_data_collator(features)
    def collect_fn_img_2d(self, features, img_aug, batch_limiter,img_size,patch_size):
        # minus batch_limiter means no batch limiter
        length_list = []
        img_list = []

        for feature in features:
            feature.pop('检查号')
            img_path = feature.pop("img_path")
            img, img_attn = self.load_img(img_path, img_aug,patch_size,img_size,False)
            feature["ground_truth"] = img.astype("float16")
            length_list.append(feature["ground_truth"].shape[1])
            img_list.append(feature["ground_truth"][0])
        ret = {"length":torch.tensor(length_list)}
        gt = np.concatenate(img_list, axis=0)[:, None, :, :]
        if batch_limiter > 0:
            rd_index = list(range(0, gt.shape[0]))
            random.shuffle(rd_index)
            rd_index = rd_index[:batch_limiter]
            gt = gt[rd_index]
        ret["ground_truth"] = torch.tensor(gt)
        return ret
        # return default_data_collator(features)
class TextProcesser():
    def __init__(self, ):
        self.control_char_re = re.compile('\s')
    def dataset_load(self, path,test_ratio):
        f_list = glob.glob(os.path.join(path, "*.xlsx"))
        e_list = []
        for f in f_list:
            e = pd.read_excel(f)
            e_list.append(e)
        e = pd.concat(e_list)
        dataset = Dataset.from_pandas(e)
        dataset = dataset.shuffle(seed=42).train_test_split(test_size=test_ratio)
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

    def collect_fn_vl(self, features, tokenizer, ip, masker, img_aug, padding_length=512):
        for feature in features:
            feature.pop("input_text")
            word_ids = feature.pop("word_ids")
            img_path = feature.pop("img_path")
            input_ids = feature.pop("input_ids")
            img, img_attn = ip.load_img(img_path, img_aug)
            masked_img, mask = masker.mask_ori_image(img.copy(), 64)  # TODO: fixed for now
            input_ids, new_labels, attn_mask = masker.whole_word_masking(input_ids, word_ids,
                                                                         tokenizer)  # (self, input_ids, word_ids, tokenizer)
            input_ids = self.list_padding(input_ids, tokenizer.pad_token_id, padding_length)
            new_labels = self.list_padding(new_labels, -100, padding_length)
            attn_mask = self.list_padding(attn_mask, 0, padding_length)
            img_attn_mask = [1] + np.reshape(img_attn, [-1]).tolist()  # TODO: assuming img cls is true
            feature["labels"] = new_labels
            feature["attention_mask"] = attn_mask + img_attn_mask
            feature["input_ids"] = input_ids
            feature["ground_truth"] = img.astype("float16")
            feature["masked_img"] = masked_img.astype("float16")
            feature["img_loss_mask"] = mask * img_attn
            feature["token_type_ids"] = [0] * padding_length + [
                1] * 513  # TODO: it is fixed in vl.assuming img cls is true
        return default_data_collator(features)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import torch
    from masker import Masker
    from matplotlib import pyplot as plt

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    # # img_tokenizer = TrilinearTokenizer()
    # # tp = TextProcesser()
    ip = ImageProcesser(norm_max=1000, norm_min=-1000,cut_incomplete_patch=False)
    ds = ip.dataset_load_img(["/dataset/nii","/dataset/nii_ori","/dataset/medical-beit/HMCT"],0.01)

    # f_dict = ip.get_file_dict("/dataset/medical-beit/HMCT")
    # dataset = tp.dataset_load(r"/dataset/medical-beit")
    # dataset = tp.dataset_preprocess(dataset, tokenizer, ip, f_dict)
    # m = Masker(0.15, 0.5)
    train_dataloader = DataLoader(ds["train"], shuffle=True, batch_size=4,
                                  collate_fn=lambda x: ip.collect_fn_img_2d(x,None))
    for i in train_dataloader:
        print(i['ground_truth'].size())
        print(i["length"],i["length"].dtype)
        plt.subplot(2,3,1)
        plt.imshow(i['ground_truth'][45,0])
        plt.subplot(2,3,2)
        plt.imshow(i['ground_truth'][78,0])
        plt.subplot(2,3,3)
        plt.imshow(i['ground_truth'][115,0])
        plt.subplot(2,3,4)
        plt.imshow(i['ground_truth'][400,0])
        plt.subplot(2,3,5)
        plt.imshow(i['ground_truth'][545,0])
        plt.subplot(2,3,6)
        plt.imshow(i['ground_truth'][1000,0])
        plt.show()
        exit()