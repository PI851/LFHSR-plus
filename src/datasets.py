import torch
import numpy as np
import torch.utils.data as data
import os
import utils_train
import math


class RandomCrop(object):
    """
        TODO 为什么需要进行这样的切割呢

        output_size:不知道什么意义
    """

    def __init__(self, output_size):
        self.output_size = (output_size, output_size)

    def __call__(self, train_data, gt_image, gt_mask_image, scale, shave=0):
        h, w = train_data.shape[-2], train_data.shape[-1]
        new_h, new_w = self.output_size

        top = np.random.randint(0 + shave, h - new_h - shave)
        left = np.random.randint(0 + shave, w - new_w - shave)

        train_data_tmp = train_data[..., top: top + new_h, left: left + new_w]

        gt_data_tmp = gt_image[..., top * scale: top * scale + new_h * scale,
                      left * scale: left * scale + new_w * scale]

        if gt_mask_image is None:
            gt_mask_image = None
        else:
            gt_mask_image = gt_mask_image[..., top * scale: top * scale + new_h * scale,
                            left * scale: left * scale + new_w * scale]

        return train_data_tmp, gt_data_tmp, gt_mask_image

class LFHSR_shear_Dataset(data.Dataset):
    def __init__(self, dir_LF,repeat=4, view_n_ori=9, view_n_input=9, scale=2, disparity_list=None, crop_size=32, if_flip=False, if_rotation=False, is_all=False):
        """
        生成数据集
        :param dir_LF: 数据路径
        :param repeat:
        :param view_n_ori: 原始光场图像的角分辨率
        :param view_n_input: 输出光场图像的角分辨率
        :param scale: 上采样比例因子
        :param disparity_list: 视差采样列表
        :param crop_size: 切割大小
        :param if_flip:
        :param if_rotation:
        :param is_all:
        """

        self.crop_size = crop_size
        self.repeat = repeat
        self.view_n_ori = view_n_ori
        self.view_n_input = view_n_input
        self.RandomCrop = RandomCrop(crop_size)
        self.if_flip = if_flip
        self.if_rotation = if_rotation
        self.is_all = is_all
        self.scale = scale
        self.dir_LF = dir_LF
        self.train_data_all = []
        self.gt_data_all = []
        self.gt_mask_data_all = []
        self.numbers = len(os.listdir(dir_LF))
        self.view_position_all = []
        self.D = len(disparity_list)

        img_list = os.listdir(dir_LF)
        # TODO 为什么这里需要进行排序呢，不排序不可以吗
        img_list.sort()
        for img_name in img_list:

            # TODO 既然所有返回的gt_mask_data都是None，为什么还要在这里定义呢
            train_data, gt_data, gt_mask_data, view_position = utils_train.data_prepare_new(dir_LF + img_name, view_n_ori,
                                                                                            view_n_input, scale,
                                                                                            disparity_list)
            self.train_data_all.append(train_data)
            self.gt_data_all.append(gt_data)
            self.gt_mask_data_all.append(gt_mask_data)
            self.view_position_all.append(view_position)

        # TODO shave参数有什么作用
        self.shave = math.ceil(max(disparity_list) * (view_n_input - 1) // 2 / scale)

    def __len__(self):
        '''
        TODO 为什么长度要乘以repeat呢，这个repeat到底代表什么意思
        :return:
        '''

        return self.repeat * self.numbers

    def __getitem__(self, idx):

        train_data = self.train_data_all[idx // self.repeat]
        gt_data = self.gt_data_all[idx // self.repeat]
        gt_mask_data = self.gt_mask_data_all[idx // self.repeat]

        train_data, gt_data, gt_mask_data = self.RandomCrop(train_data, gt_data, gt_mask_data, self.scale, self.shave)

        if self.if_flip:
            if np.random.rand(1) >= 0.5:
                train_data = np.flip(train_data, 3)
                train_data = np.flip(train_data, 1)

                gt_data = np.flip(gt_data, 2)
                gt_data = np.flip(gt_data, 0)

                if gt_mask_data is not None:
                    gt_mask_data = np.flip(gt_mask_data, 1)

        if self.if_rotation:
            k = np.random.randint(0, 4)
            train_data = np.rot90(train_data, k, (3, 4))
            train_data = np.rot90(train_data, k, (1, 2))

            gt_data = np.rot90(gt_data, k, (2, 3))
            gt_data = np.rot90(gt_data, k, (0, 1))

            if gt_mask_data is not None:
                gt_mask_data = np.rot90(gt_mask_data, k, (1, 2))

        if gt_mask_data is None:
            gt_mask_data = np.array([-1])

        if self.is_all:
            return torch.from_numpy(train_data.copy()), \
                   torch.from_numpy(train_data[self.D // 2].copy()), \
                   torch.from_numpy(gt_data[self.view_position_all[idx // self.repeat][0]
                   , self.view_position_all[idx // self.repeat][1]].copy()), \
                   self.view_position_all[idx // self.repeat], \
                   torch.from_numpy(gt_data.copy()), \
                   torch.from_numpy(gt_mask_data.copy())
        else:
            view_u = np.random.randint(0, self.view_n_input)
            view_v = np.random.randint(0, self.view_n_input)
            view_position = (view_u, view_v)

            return torch.from_numpy(train_data.copy()), \
                   torch.from_numpy(train_data[self.D // 2, view_u, view_v].copy()), \
                   torch.from_numpy(gt_data[self.view_n_input // 2, self.view_n_input // 2].copy()), \
                   view_position, \
                   torch.from_numpy(gt_data[view_u, view_v].copy())