import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from random import randint

def weight_L1Loss(X, Y):
    weight_error = torch.abs(X - Y)
    loss = torch.sum(weight_error) / torch.numel(weight_error)
    return loss

def warp_all(img,  disparity_list, view_n, view_position, align_corners=False):
    """
    1.0理解：将所有的图像朝中心视图进行扭曲
    :param img: 图像数据
    :param disparity_list: 视差值采样列表
    :param view_n: 角度分辨率
    :param view_position: 高分辨视图的位置
    :param align_corners:
    :return:
    """
    B, C, UV, X, Y = list(img.shape)
    D = len(disparity_list)
    img = img.permute(2, 0, 1, 3, 4).reshape(UV, B * C, X, Y)
    img_all = []
    target_position = np.array([view_position[0], view_position[1]])

    for disparity in disparity_list:
        theta = []
        for i in range(view_n):
            for j in range(view_n):
                ref_position = np.array([i, j])
                d = (target_position - ref_position) * disparity * 2
                # TODO 具体底层是怎么实现warp的呢
                theta_t = torch.FloatTensor([[1, 0, d[1] / img.shape[3]], [0, 1, d[0] / img.shape[2]]])
                theta.append(theta_t.unsqueeze(0))
        theta = torch.cat(theta, 0).cuda()
        grid = F.affine_grid(theta, img.size(), align_corners=align_corners)
        img_tmp = F.grid_sample(img, grid, align_corners=align_corners)
        img_tmp.unsqueeze(0)
        img_all.append(img_tmp)
    img_all = torch.cat(img_all, 0)
    img_all = img_all.reshape(D, UV, B, C, X, Y).permute(2, 3, 0, 1, 4, 5)  # B, C, D, UV, X, Y
    return img_all


def data_prepare_new(dir_LF, view_n_ori, view_n_new, scale, disparity_list):
    """
    1.0理解：生成训练数据集，为什么需要对原始数据进行下采样作为训练数据，原始数据的空间分辨率都是相同的
    所以是不是说原始图像的分辨率就是目标达到的分辨率，只是自己构造了低分辨率的数据
    为什么是1，1，-1。。。是因为通道数不确定吗
    为什么变成低分辨率视图之后，视差范围就会变小呢

    2.0理解：自己构造的训练数据，所以需要对原始图像进行下采样，-1代表不确定
    :param dir_LF: 原始光场图像的路径
    :param view_n_ori: 原始光场图像的角分辨率
    :param view_n_new: 输出光场图像的角分辨率
    :param scale: 上采样比例因子
    :param disparity_list: 视差值列表
    :return:
    """

    # 保证输出的光场图像的角分辨率为奇数
    assert view_n_new % 2 == 1
    D = len(disparity_list)
    gt_y = image_prepare_npy(dir_LF, view_n_ori, view_n_new)
    U, V, X, Y = list(gt_y.shape)
    # 生成训练数据的分辨率，下采样
    lr_X = X // scale
    lr_Y = Y // scale
    X = lr_X * scale
    Y = lr_Y * scale
    lr_y = np.zeros((view_n_new, view_n_new, lr_X, lr_Y), dtype=np.float32)
    # 监督数据
    gt_y = gt_y[..., :X, :Y]
    # 随机生成高分辨视图的位置
    view_position = [randint(0, view_n_new - 1), randint(0, view_n_new - 1)]

    # 构造训练数据
    for i in range(view_n_new):
        for j in range(view_n_new):
            img = Image.fromarray(gt_y[i, j] / 255.0)
            # 对原始光场图像进行下采样，生成低分辨率的图像
            img_tmp = img.resize((lr_X, lr_Y), Image.BICUBIC)
            lr_y[i, j, ...] = img_tmp
    # TODO 为什么要进行这样的维度转化
    lr_y = torch.from_numpy(lr_y.copy()).cuda().reshape(1, 1, -1, lr_X, lr_Y)

    # TODO 为什么低分辨视图的视差范围会变小，感觉公式里和视差有关的变量没有改变
    # lr_y_sheared = warp_all(lr_y, disparity_list / scale, view_n_new, view_position=[view_n_new // 2, view_n_new // 2])
    lr_y_sheared = warp_all(lr_y, disparity_list / scale, view_n_new, view_position=view_position)
    lr_y_sheared = lr_y_sheared.reshape(D, U, V, lr_X, lr_Y).cpu().numpy()
    gt_y /= 255.0

    return lr_y_sheared, gt_y, None, view_position

def angular_resolution_changes(image, view_num_ori, view_num_new):
    """
    对光场图像的角分辨率进行改变
    但是为什么view_num_ori需要+1呢，是为了保证向下裁剪吗
    :param image: 输入的光场图像
    :param view_num_ori: 原始光场图像的角分辨率
    :param view_num_new: 超分辨输出的光场图像的角分辨率
    :return: image: 裁剪之后的光场图像
    """

    n_view = (view_num_ori + 1 - view_num_new) // 2
    return image[n_view:n_view + view_num_new, n_view:n_view + view_num_new, :, :]

def image_prepare_npy(image_path, view_n_ori, view_n_new):
    """
    读取npy格式的光场图像
    :param image_path: 数据路径
    :param view_n_ori: 原始光场图像的角分辨率
    :param view_n_new: 超分辨输出的光场图像的角分辨率
    :return: gt_image_input: 处理之后的光场图像
    """

    gt_image = np.load(image_path)
    # 转换数据类型
    gt_image = gt_image.astype(np.float32)
    # 所需的角分辨率小于原始角分辨率
    if view_n_new < view_n_ori:
        gt_image_input = angular_resolution_changes(gt_image, view_n_ori, view_n_new)
    else:
        gt_image_input = gt_image

    return gt_image_input

def get_parameter_number(net):
    print(net)
    parameter_list = [p.numel() for p in net.parameters()]
    print(parameter_list)
    total_num = sum(parameter_list)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
