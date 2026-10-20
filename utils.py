"""
Utils for Dataset
"""
import random
import torch
import numpy as np
import operator
import os
import logging
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CLASS_LABELS = {
    'CHAOST2': {
        'pa_all': set(range(1, 5)),
        0: set([1, 4]),  # upper_abdomen, leaving kidneies as testing classes
        1: set([2, 3]),  # lower_abdomen
    },
}


def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 0
    return fg_bbox, bg_bbox


def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()


def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)


class Scores():

    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.patient_dice = []
        self.patient_iou = []

    def record(self, preds, label):
        assert len(torch.unique(preds)) < 3

        tp = torch.sum((label == 1) * (preds == 1))
        tn = torch.sum((label == 0) * (preds == 0))
        fp = torch.sum((label == 0) * (preds == 1))
        fn = torch.sum((label == 1) * (preds == 0))

        self.patient_dice.append(2 * tp / (2 * tp + fp + fn))
        self.patient_iou.append(tp / (tp + fp + fn))

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn

    def compute_dice(self):
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def compute_iou(self):
        return self.TP / (self.TP + self.FP + self.FN)


def set_logger(path):
    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter('[%(levelname)] - %(name)s - %(message)s')
    logger.setLevel("INFO")

    # log to .txt
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger



class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, net_output, target, weight=None, sigmoid=False):
        if sigmoid:
            net_output = torch.sigmoid(net_output) # b 1 h w
        assert net_output.size() == target.size(), 'predict {} & target {} shape do not match'.format(net_output.size(), target.size())
        # print(net_output[:, 0].shape, target[:, 0].shape) #torch.Size([1, 256, 256]) torch.Size([1, 256, 256])
        dice_loss = self._dice_loss(net_output[:, 0], target[:, 0])
        return dice_loss


class Mask_DC_and_BCE_loss(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_loss, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target.float())
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss


def get_criterion():


    pos_weight = torch.ones([1]).cuda()*2

    criterion = Mask_DC_and_BCE_loss(pos_weight=pos_weight)

    return criterion


class CrossArchitectureKDLoss(nn.Module):

    def __init__(self, T=2.0):
        super(CrossArchitectureKDLoss, self).__init__()
        self.T = T

    def forward(self, cnn_logits, sam_logits, direction="sam_to_cnn"):

        cnn_log_prob = F.log_softmax(cnn_logits / self.T, dim=1)
        cnn_prob = F.softmax(cnn_logits / self.T, dim=1)

        sam_fg_prob = torch.sigmoid(sam_logits / self.T)  # [B, 1, H, W]
        sam_bg_prob = 1.0 - sam_fg_prob  # [B, 1, H, W]

        sam_prob = torch.cat([sam_bg_prob, sam_fg_prob], dim=1)  # [B, 2, H, W]

        sam_log_prob = torch.log(sam_prob + 1e-8)

        if direction == "sam_to_cnn":
            loss = F.kl_div(cnn_log_prob, sam_prob.detach(), reduction="none")
        elif direction == "cnn_to_sam":
            loss = F.kl_div(sam_log_prob, cnn_prob.detach(), reduction="none")
        else:
            raise ValueError("Invalid direction")

        loss = loss.sum(dim=1).mean() * (self.T ** 2)

        return loss
