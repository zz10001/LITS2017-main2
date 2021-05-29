# -*- coding: utf-8 -*-

import os
import numpy as np
import SimpleITK as sitk
# -*- coding: utf-8 -*-
from time import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import datetime
import scipy.ndimage as ndimage
from skimage.transform import resize

import numpy as np
from tqdm import tqdm

from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from net import UNet,VNet
from utils.metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
from utils.utils import *
from sklearn.externals import joblib

import imageio
import sys
sys.path.append('../utils')
from utils import *

test_ct_path = '../LITS2017/LITS2017_test'
test_seg_path = '../LITS2017/test_liver'   #在70个样例分割好的肝脏标签

pred_path = 'pred_test'
BLOCKSIZE = (64, 128, 128) #每个分块的大小
new_spacing = [0.8, 0.8, 1.5]

if not os.path.exists(pred_path):
    os.mkdir(pred_path)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='LITS',
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='')
    parser.add_argument('--patchsize', default=(64,128,160), type=int, metavar='N',
                        help='number of slice')

    args = parser.parse_args()

    return args

def generate_test_locations(image, patch_size, stride):
    # 40-128-160, liver_patch shape
    ww,hh,dd = image.shape
    print('image.shape',ww,hh,dd)
    sz = math.ceil((ww - patch_size[0]) / stride[0]) + 1
    sx = math.ceil((hh - patch_size[1]) / stride[1]) + 1
    sy = math.ceil((dd - patch_size[2]) / stride[2]) + 1

    return (sz,sx,sy),(ww,hh,dd)

def infer_tumorandliver(model,ct_array_nor,pad_p, tmp, cube_shape=(64,128,128),pred_threshold=0.6):
    patch_size = cube_shape
    patch_stride = [16,32,32]

    mask_pred_containers = np.zeros((ct_array_nor.shape)).astype(np.float32)   #用来存放结果
    locations,image_shape = generate_test_locations(ct_array_nor, patch_size, patch_stride)  #32 64 80
    print('location',locations,image_shape)

    seg_liver = np.zeros((1, )+(ct_array_nor.shape)).astype(np.float32)
    cnt_liver = np.zeros((ct_array_nor.shape)).astype(np.float32)

    seg_tumor = np.zeros((1, )+(ct_array_nor.shape)).astype(np.float32)
    cnt_tumor = np.zeros((ct_array_nor.shape)).astype(np.float32)

    print('seg_liver shape',seg_liver.shape)

    print('   ')

    for z in range(0,locations[0]):
        zs =  min(patch_stride[0]*z, image_shape[0]-patch_size[0])
        for x in range(0,locations[1]):
            xs = min(patch_stride[1]*x, image_shape[1]-patch_size[1])
            for y in range(0,locations[2]):
                ys = min(patch_stride[2]*y, image_shape[2]-patch_size[2])
                
                patch = ct_array_nor[zs:zs + patch_size[0], 
                                 xs:xs + patch_size[1], 
                                 ys:ys + patch_size[2]]
                # print('patch',patch)
                patch = np.expand_dims(np.expand_dims(patch,axis=0),axis=0).astype(np.float32)
                patch_tensor = torch.from_numpy(patch).cuda()

                output = model(patch_tensor)
                output = torch.sigmoid(output)
                output = output.cpu().data.numpy()
                # print('output',output.shape)
                seg_liver[:, zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] \
                    = seg_liver[:,zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] + output[0,0,:,:,:]
            
                cnt_liver[zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] \
                    = cnt_liver[zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] + 1

                seg_tumor[:, zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] \
                    = seg_tumor[:,zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] + output[0,1,:,:,:]
            
                cnt_tumor[zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] \
                    = cnt_tumor[zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] + 1

    seg_liver = seg_liver / np.expand_dims(cnt_liver,axis=0)
    seg_tumor = seg_tumor / np.expand_dims(cnt_tumor,axis=0)

    seg_liver = np.squeeze(seg_liver)
    seg_tumor = np.squeeze(seg_tumor)

    seg_liver = np.where(seg_liver>=pred_threshold, 1, 0)
    seg_tumor = np.where(seg_tumor>=pred_threshold, 2, 0)

    mask_pred_containers = seg_tumor + seg_liver
    mask_pred_containers[mask_pred_containers>1]=2

    # 弄回pad前大小
    return mask_pred_containers[pad_p[0][0]:pad_p[0][0]+tmp[0],
                                pad_p[1][0]:pad_p[1][0]+tmp[1],
                                pad_p[2][0]:pad_p[2][0]+tmp[2]]
                                
def normalize(slice, bottom=99.5, down=0.5):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9
        return tmp

def get_realfactor(spa,xyz,ct_array):
    resize_factor = spa / xyz
    print('resize',resize_factor)
    new_real_shape = ct_array.shape * resize_factor[::-1]
    # print('new',new_real_shape)
    new_shape = np.round(new_real_shape)
    print('new',new_real_shape)
    real_resize_factor = new_shape / ct_array.shape
    rezoom_factor = ct_array.shape / new_shape
    return real_resize_factor, rezoom_factor

# 用于将得到的肝脏区域大小pad成训练时候patch大小的倍数
def make_patch(image, patch_size=(64,128,160)):
    w, h, d = image.shape
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=-9)
    return image,[[wl_pad,wr_pad],[hl_pad,hr_pad],[dl_pad,dr_pad]]

def main():
    args = parse_args()

    # create model
    print("=> creating model %s" %args.arch)
    model = UNet.UNet3d(in_channels=1, n_classes=2, n_channels=32)
    model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load('models/LITS_UNet_lym/2021-05-04-22-53-45/epoch143-0.9682-0.8594_model.pth'))
    model.eval()
    #model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    
    for file_index, file in enumerate(os.listdir(test_ct_path)):
        start_time = time()
        # 将要预测的CT读入
        ct = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(test_seg_path,file.replace('volume', 'segmentation').replace('nii','nii.gz')), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        if file.replace('volume', 'segmentation').replace('nii', 'nii.gz') in os.listdir(pred_path):
            print('already predict {}'.format(file))
            continue

        print('start predict file:',file,ct_array.shape)

        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        spacing = np.array(list(ct.GetSpacing()))
        print('-------',file,'-------')
        print('original space', np.array(ct.GetSpacing()))
        print('original shape and spacing:',ct_array.shape, spacing)

        # step1: spacing interpolation

        real_resize_factor, rezoom_factor = get_realfactor(spacing,new_spacing,ct_array)
        # 根据输出out_spacing设置新的size
        ct_array_zoom = ndimage.zoom(ct_array, real_resize_factor, order=3)
        seg_array_zoom = ndimage.zoom(seg_array, real_resize_factor, order=0)
        slice_predictions = np.zeros((ct_array_zoom.shape),dtype=np.int16)  #zoom之后大小,裁剪肝脏区域前的大小
        # 对金标准插值不应该使用高级插值方式，这样会破坏边界部分,检查数据输出很重要！！！
        
        # step2 :get mask effective range(startpostion:endpostion)
        pred_liver = seg_array_zoom.copy()
        pred_liver[pred_liver>0] = 1
        bb = find_bb(pred_liver)
        ct_array_zoom = ct_array_zoom[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]]
        print('effective shape or before pad shape:', ct_array.shape,',',seg_array.shape)

        # step3:标准化Normalization
        ct_array_nor = normalize(ct_array_zoom)
        w, h, d = ct_array_nor.shape
        tmp = [w, h, d]  #这个为pad前的肝脏区域 即bb范围大小
        # 然后为了能够更好的进行预测，将需要测试的图像shape大小弄成patch的倍数，以便能够进行滑窗预测
        ct_array_nor, pad_p = make_patch(ct_array_nor,patch_size=[64,128,160])
        print('after pad shape', ct_array_nor.shape)

        # 开始预测
        pred_seg = infer_tumorandliver(model,ct_array_nor,pad_p, tmp, cube_shape=(64,128,160), pred_threshold=0.6)
        print('after infer shape', pred_seg.shape)  #大小为crop出来的肝脏区域大小
        slice_predictions[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] = pred_seg

        # 恢复到原始尺寸大小
        slice_predictions = ndimage.zoom(slice_predictions,rezoom_factor,order=0)
        slice_predictions = slice_predictions.astype(np.uint8)
        print('slice_predictions shape',slice_predictions.shape)

        predict_seg = sitk.GetImageFromArray(slice_predictions)
        predict_seg.SetDirection(ct.GetDirection())
        predict_seg.SetOrigin(ct.GetOrigin())
        predict_seg.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(predict_seg, os.path.join(pred_path, file.replace('volume', 'segmentation').replace('nii', 'nii.gz')))

        speed = time() - start_time

        print(file, 'this case use {:.3f} s'.format(speed))
        print('-----------------------')

        torch.cuda.empty_cache()
                    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()