import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        inputs = torch.sigmoid(input)
        num = target.size(0)
        input_1 = inputs[:,0,:,:,:]
        input_2 = inputs[:,1,:,:,:]

        target_1 = target[:,0,:,:,:]
        target_2 = target[:,1,:,:,:]

        input_1 = input_1.view(num, -1)
        target_1 = target_1.view(num, -1)

        input_2 = input_2.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)

        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num
        dice_2 = 1 - dice_2.sum() / num

        dice = dice_1*0.4+dice_2*0.6
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
