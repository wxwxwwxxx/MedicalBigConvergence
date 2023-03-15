import numpy as np
import torch
import matplotlib.pyplot as plt


class Masker():
    def __init__(self):
        pass

    def mask_ori_image(self, img, patch_size,mask_prob_img):
        # image resolution: 512 512 512
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


if __name__ == "__main__":
    m = Masker()
    input = np.ones([1, 128,128,128])
    output = m.mask_ori_image(input, 16,0.85)
    plt.imshow(output[0][0, :, :, 0])
    plt.show()
    print(output.shape)
