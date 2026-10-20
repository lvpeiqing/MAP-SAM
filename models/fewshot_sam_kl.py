"""
ablation of lambda
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.sam.build_sam import sam_model_registry
from .discriminator import Discriminator
import random
import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 强制使用无界面的后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label
from random import randint


class FewShotSegSAM(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        # SAM
        self.device = torch.device('cuda')
        self.sam_model = sam_model_registry[args['model_type']](args, pixel_mean=[123.675, 116.28, 103.53],
                                                                pixel_std=[58.395, 57.12, 57.375]).to(self.device).train()
        self.Discriminator = Discriminator(in_channels=1, out_conv_channels=2).to(self.device)

        self.dilation_radius = 1

        for n, value in self.sam_model.image_encoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True
        for n, value in self.sam_model.prompt_encoder.named_parameters():
            value.requires_grad = True

        for n, value in self.sam_model.mask_decoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True


    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, query_pred,  isval):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            qry_mask: query mask
                1 x H x W  tensor
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                               + [torch.cat(qry_imgs, dim=0), ], dim=0)

        # ########################################################
        query_pred_cnn = query_pred.argmax(dim=1)  # C x H x W
        query_pred_cnn = query_pred_cnn.detach().cpu().numpy()
        mask_cls = (query_pred_cnn[0] == 1).astype(np.uint8)

        label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)

        if region_ids == 0:
            print('111111111')
            box = torch.tensor([[0.0, 0.0, 255.0, 255.0]], device=self.device, dtype=torch.float)
            box_np = np.array([0.0, 0.0, 255.0, 255.0], dtype=np.float32)

            empty_pts = np.zeros((0, 2), dtype=np.int32)
            pos_points = None
            neg_points = None
            K = None
            maps = None
            coords_torch, labels_torch = self.format_for_sam(empty_pts, empty_pts)
            pt = (coords_torch, labels_torch)

        else:
            ratio_list, regionid_list = [], []
            for region_id in range(1, region_ids + 1):
                binary_msk = np.where(label_msk == region_id, 1, 0)
                r = np.sum(binary_msk) / np.sum(mask_cls)
                ratio_list.append(r)
                regionid_list.append(region_id)

            ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
            regionid_list = regionid_list[::-1]

            binary_msk = np.where(label_msk == regionid_list[0], 1, 0)
            query_pred_cnn_np = binary_msk

            box, box_np = self.fixed_bbox(query_pred_cnn_np) if isval else self.random_bbox(query_pred_cnn_np)

            pos_points, neg_points, K, maps, R_gradient  = self.generate_prompts_from_mask(query_pred_cnn_np, s=25)
            coords_torch, labels_torch = self.format_for_sam(pos_points, neg_points)
            pt = (coords_torch, labels_torch)

        ####################################################
        query_pred_detached = query_pred.detach()

        if region_ids == 0:
            fg_prob = query_pred_detached[:, 1:2, ...]
            mask_logits = torch.logit(fg_prob, eps=1e-5)
            uncertainty_area = None
            R_gradient = None
            fusion_map = None

        else:
            fusion_map, uncertainty_area, R_gradient = self.Discriminator(query_pred_detached, R_gradient)
            mask_logits = fusion_map[:, 1:2, ...]

        #####################################################

        imgs_concat = self.denorm(imgs_concat)
        imgs_concat = self.sam_model.preprocess(imgs_concat)
        image_embeddings = self.sam_model.image_encoder(imgs_concat)

        qry_fts = image_embeddings[self.n_ways * self.n_shots * supp_bs:]  # torch.Size([1, 256, 16, 16])

        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
               points=pt,
               boxes=box,
               masks=F.interpolate(mask_logits, size=(64, 64), mode='bilinear')
             )

        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
             image_embeddings=qry_fts,
             image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
             sparse_prompt_embeddings=sparse_embeddings,
             dense_prompt_embeddings=dense_embeddings,
             multimask_output=self.args['multimask'],  # False
          )

        pred_sam = F.interpolate(low_res_masks, size=(self.args['image_size'], self.args['image_size']), mode="bilinear",
                                 align_corners=False)  # torch.Size([1, 1, 256, 256])

        return pred_sam,region_ids,pos_points, neg_points,K, maps,box_np, uncertainty_area, R_gradient, fusion_map


    def denorm(self, img):

        img = img.cpu().detach().numpy()

        min = np.min(img)
        max = np.max(img)
        img = (img - (min)) / (max - min) * 255
        img = img.astype(np.uint8)

        img = torch.from_numpy(img).float().to(self.device)

        return img

    def _create_circular_kernel(self, radius):
        """Create circular structuring element."""
        diameter = 2 * radius + 1 #3:7；1:3；
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        return kernel

    def farthest_point_sampling(self,points, k):
        n = len(points)
        if n == 0: return np.zeros((0, 2), dtype=np.int32)
        if n <= k: return points.copy()

        sampled = [points[0]]
        dist = np.sum((points - sampled[0]) ** 2, axis=1)

        for _ in range(1, k):
            idx = np.argmax(dist)
            sampled.append(points[idx])
            new_dist = np.sum((points - points[idx]) ** 2, axis=1)
            dist = np.minimum(dist, new_dist)

        return np.array(sampled, dtype=np.int32)

    def get_max_distance_point(self,binary_mask):
        if binary_mask.sum() == 0: return None
        dist_map = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(dist_map), dist_map.shape)
        return np.array([y, x], dtype=np.int32)

    def generate_prompts_from_mask(self, label, erode_iter=2, dilate_iter=2, s=25, K_min=4, K_max=8):
        if label.ndim == 3: label = label[0, ...]

        label_bin = label.astype(np.uint8)
        kernel = self._create_circular_kernel(self.dilation_radius)
        M_erode_core = cv2.erode(label_bin, kernel, iterations=erode_iter)
        M_dilate_full = cv2.dilate(label_bin, kernel, iterations=dilate_iter)

        R_in = label_bin - M_erode_core
        R_out = M_dilate_full - label_bin
        R_gradient = M_dilate_full - M_erode_core

        contours, _ = cv2.findContours(label_bin * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_len = sum(cv2.arcLength(c, True) for c in contours) if len(contours) > 0 else 0
        K = max(K_min, min(math.ceil(contour_len / s), K_max)) if contour_len > 0 else K_min

        if M_erode_core.sum() > 0:
            core_point = self.get_max_distance_point(M_erode_core)
            pts_in = np.stack(np.where(R_in > 0), axis=1)

        else:
            core_point = self.get_max_distance_point(label_bin)
            pts_in = np.stack(np.where(label_bin > 0), axis=1)

        pos_boundary_points = self.farthest_point_sampling(pts_in, K) if len(pts_in) > 0 else np.zeros(
            (0, 2))
        pos_points = np.concatenate([core_point[None, :], pos_boundary_points],
                                    axis=0) if core_point is not None else pos_boundary_points

        pts_out = np.stack(np.where(R_out > 0), axis=1)
        neg_points = self.farthest_point_sampling(pts_out, K) if len(pts_out) > 0 else np.zeros((0, 2))

        maps = {"mask": label_bin, "R_in": R_in, "R_out": R_out}
        return pos_points, neg_points, K, maps,R_gradient

    # ==========================================
    def format_for_sam(self, pos_points, neg_points):

        pos_xy = pos_points[:, [1, 0]] if len(pos_points) > 0 else np.empty((0, 2))
        neg_xy = neg_points[:, [1, 0]] if len(neg_points) > 0 else np.empty((0, 2))

        sam_coords = np.concatenate([pos_xy, neg_xy], axis=0)

        pos_labels = np.ones(len(pos_xy), dtype=np.int32)
        neg_labels = np.zeros(len(neg_xy), dtype=np.int32)
        sam_labels = np.concatenate([pos_labels, neg_labels], axis=0)

        coords_torch = torch.as_tensor(sam_coords, dtype=torch.float32, device=self.device)
        labels_torch = torch.as_tensor(sam_labels, dtype=torch.int, device=self.device)

        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        return coords_torch, labels_torch

    def random_bbox(self, mask, class_id=1, img_size=256):

        indices = np.argwhere(mask == class_id)
        indices[:, [0, 1]] = indices[:, [1, 0]]

        minx = np.min(indices[:, 0])
        maxx = np.max(indices[:, 0])
        miny = np.min(indices[:, 1])
        maxy = np.max(indices[:, 1])

        classw_size = maxx - minx + 1
        classh_size = maxy - miny + 1

        shiftw = randint(int(0.95 * classw_size), int(1.05 * classw_size))
        shifth = randint(int(0.95 * classh_size), int(1.05 * classh_size))
        shiftx = randint(-int(0.05 * classw_size), int(0.05 * classw_size))
        shifty = randint(-int(0.05 * classh_size), int(0.05 * classh_size))

        new_centerx = (minx + maxx) // 2 + shiftx
        new_centery = (miny + maxy) // 2 + shifty

        minx = np.max([new_centerx - shiftw // 2, 0])
        maxx = np.min([new_centerx + shiftw // 2, img_size - 1])
        miny = np.max([new_centery - shifth // 2, 0])
        maxy = np.min([new_centery + shifth // 2, img_size - 1])

        box = torch.tensor([[minx, miny, maxx, maxy]],
                     device=self.device, dtype=torch.float)

        box_np = np.array([minx, miny, maxx, maxy], dtype=np.float32)

        return box, box_np

    def fixed_bbox(self, mask, class_id=1, img_size=256):
        indices = np.argwhere(mask == class_id)  # Y X (0, 1)
        indices[:, [0, 1]] = indices[:, [1, 0]]

        minx = np.min(indices[:, 0])
        maxx = np.max(indices[:, 0])
        miny = np.min(indices[:, 1])
        maxy = np.max(indices[:, 1])

        box = torch.tensor([[minx, miny, maxx, maxy]],
                           device=self.device, dtype=torch.float)

        box_np = np.array([minx, miny, maxx, maxy], dtype=np.float32)

        return box, box_np


def add_bbox_patch(ax, bbox):
    if bbox is not None and len(bbox) == 4 and sum(bbox) > 0:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2.5, edgecolor='dodgerblue', facecolor='none',
                                 linestyle='--', label='BBox')
        ax.add_patch(rect)


def visualize_five_panels(image, maps, pos_points, neg_points, K,bbox,image_path, i_iter):

    mask = maps["mask"]
    R_in = maps["R_in"]
    R_out = maps["R_out"]
    H, W = mask.shape

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Boundary-Guided Prompt Sampling (K={K})", fontsize=16)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Boundary-Guided Prompt Sampling (Points & Box)", fontsize=18, fontweight='bold')

    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title("1. Original Mask")
    axes[0].axis('off')
    axes[1].imshow(R_in, cmap='hot')
    axes[1].set_title("2. Inner Ring")
    axes[1].axis('off')
    axes[2].imshow(R_out, cmap='ocean')
    axes[2].set_title("3. Outer Ring")
    axes[2].axis('off')


    axes[3].imshow(image, cmap='gray', alpha=0.6)
    axes[3].set_title("4. Prompt Overlay")
    axes[3].axis('off')


    axes[4].set_title("5. Prompts Only")
    axes[4].set_xlim(0, W)
    axes[4].set_ylim(H, 0)
    axes[4].set_aspect('equal')
    axes[4].grid(True, linestyle=':', alpha=0.6)

    for ax in [axes[3], axes[4]]:
        if len(pos_points) > 0:
            ax.scatter(pos_points[0, 1], pos_points[0, 0], c='lime', marker='*', s=180, edgecolors='black',
                       label='Core Pos Point')
            if len(pos_points) > 1:
                ax.scatter(pos_points[1:, 1], pos_points[1:, 0], c='green', marker='o', s=45, edgecolors='white',
                           label='Inner Pos Points')

        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 1], neg_points[:, 0], c='red', marker='X', s=55, edgecolors='white',
                       label='Outer Neg Points')

        add_bbox_patch(ax, bbox)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

    save_path = os.path.join(image_path, '{}_point_box_sam.jpg'.format(i_iter))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.clf()
    plt.close('all')


def denorm(img):

    img =img.cpu().detach().numpy()

    min = np.min(img)
    max = np.max(img)
    img = (img - (min)) / (max - min) * 255
    img= img.astype(np.uint8)
    img = torch.from_numpy(img).float().cuda()

    return img






