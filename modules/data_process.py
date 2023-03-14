import SimpleITK as sitk
import numpy as np
import math


class ImageProcesser():
    def __init__(self):
        pass

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

                img = img[:, n_start:nh+n_start, :, :]
                h = nh

        if w < target_size:
            if cut_incomplete_patch:
                nw = (w // patch_size) * patch_size
                n_start = (w - nw) // 2
                img = img[:,:,  n_start:nw+n_start, :]
                w = nw

        if d < target_size:
            if cut_incomplete_patch:
                nd = (d // patch_size) * patch_size
                n_start = (d - nd) // 2
                img = img[:,:, :,  n_start:nd+n_start]
                d = nd

        attn_h = math.ceil(h / patch_size)
        attn_w = math.ceil(w / patch_size)
        attn_d = math.ceil(d / patch_size)
        attn_target = target_size // patch_size
        attn_mask = np.ones([attn_h, attn_w, attn_d])

        img = np.pad(img, [(0, 0), (0, target_size - h), (0, target_size - w), (0, target_size - d)], mode='constant')
        attn_mask = np.pad(attn_mask , [(0, attn_target - attn_h), (0, attn_target - attn_w), (0,  attn_target - attn_d)], mode='constant')
        return img, attn_mask


if __name__ == "__main__":
    ip = ImageProcesser()


    img = sitk.ReadImage("/dataset/LUNA2016/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.979083010707182900091062408058.mhd")
    img = sitk.GetArrayFromImage(img)
    img = img[None,:,:,:]
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    print(img.shape)
    ret, attn_mask = ip.padding(img, 64, 512, True)
    print(ret.shape)
    print(attn_mask)