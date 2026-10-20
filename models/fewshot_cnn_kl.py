"""
ablation of lambda
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import Res101Encoder
import matplotlib.pyplot as plt
import random
np.set_printoptions(threshold = 1e6)


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.layer_weights = nn.Parameter(torch.ones(2))
        self.branch_weights = nn.Parameter(torch.ones(2))

        self.rate = 0.95
        self.kernel = (16, 16)
        self.stride = 8


    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=30):
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
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W # supp_mask: (1, 1, 1, 256, 256)

        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao, gamma_1, gamma_2 = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        fts_size = qry_fts[1].shape[-2:]  ##torch.Size([32, 32])

        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  # t for support features
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        layer_weights = torch.softmax(self.layer_weights, dim=0)
        branch_weights = torch.softmax(self.branch_weights, dim=0)

        align_loss = torch.zeros(1).to(self.device)

        outputs = []
        mask_list = []
        final_supp_list = []

        for epi in range(supp_bs):

            # Extract prototypes
            supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in range(len(supp_fts))]

            fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))]  # prototype for support

            if supp_mask[epi][0].sum() == 0:
                fg_local_prototypes = [[[self.scan(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot], self.kernel, self.stride, flag=0)
                for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in range(len(supp_fts))]
            else:
                fg_local_prototypes = [[[self.scan(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot],self.kernel, self.stride, flag=1)
                                         for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in range(len(supp_fts))]

            supp_local_proto = [[[torch.mean(fg_local_prototypes[n][way][shot], dim=1) for shot in range(self.n_shots)]
                 for way in range(self.n_ways)] for n in range(len(supp_fts))]  # (1, C)

            qry_pred_supp = [torch.stack(
                [self.getPred(qry_fts[n][epi], supp_local_proto[n][way][epi], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'

            qry_pred_up_supp = [F.interpolate(qry_pred_supp[n], size=img_size, mode='bilinear', align_corners=True)
                           for n in range(len(qry_fts))]

            pred_supp = [layer_weights[n] * qry_pred_up_supp[n] for n in range(len(qry_fts))]
            preds_supp = torch.sum(torch.stack(pred_supp, dim=0), dim=0)

            qry_pred_coarse = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
            qry_pred_up_coarse = [F.interpolate(qry_pred_coarse[n], size=img_size, mode='bilinear', align_corners=True)
                                  for n in range(len(qry_fts))]

            pred_coarse = [layer_weights[n] * qry_pred_up_coarse[n] for n in range(len(qry_fts))]
            preds_coarse = torch.sum(torch.stack(pred_coarse, dim=0), dim=0)

            preds = branch_weights[0] * preds_supp + branch_weights[1] * preds_coarse
            preds = torch.cat((1.0 - preds, preds), dim=1)
            outputs.append(preds)

            # Prototype alignment loss
            if train:
                align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                preds, supp_mask[epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W #torch.Size([1, 1, 2, 256, 256])
        output = output.view(-1, *output.shape[2:]) #torch.Size([1, 2, 256, 256])

        return output, align_loss / supp_bs, gamma_1, gamma_2


    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C


        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes


    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        layer_weights = torch.softmax(self.layer_weights, dim=0)
        branch_weights = torch.softmax(self.branch_weights, dim=0)

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):

                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                if pred_mask[way + 1][0].sum() == 0:
                    qry_fg_local_prototypes = [[self.scan(qry_fts[n], pred_mask[way + 1], self.kernel, self.stride, flag=0)] for n in range(len(qry_fts))]
                else:
                    qry_fg_local_prototypes = [[self.scan(qry_fts[n], pred_mask[way + 1], self.kernel, self.stride, flag=1)] for n in range(len(qry_fts))]

                qry_local_proto = [[torch.mean(qry_fg_local_prototypes[n][way], dim=1) for way in range(self.n_ways)] for n in range(len(supp_fts))]  # (1, C)

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], qry_local_proto[n][way], self.thresh_pred_[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                             for n in range(len(supp_fts))]

                preds_supp = [layer_weights[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds_supp = torch.sum(torch.stack(preds_supp, dim=0), dim=0)

                supp_pred_coarse  = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred_[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred_coarse  = [F.interpolate(supp_pred_coarse [n][None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                             for n in range(len(supp_fts))]

                supp_preds_coarse= [layer_weights[n] * supp_pred_coarse[n] for n in range(len(supp_fts))]
                supp_preds_coarse = torch.sum(torch.stack(supp_preds_coarse, dim=0), dim=0)

                preds = branch_weights[0] * preds_supp + branch_weights[1] * supp_preds_coarse
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss


    def proto_alignLoss(self, supp_fts, qry_fts, pred, fore_mask, supp_prototypes):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss_sim = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                # 查询原型
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Combine prototypes from different scales
                fg_prototypes = [self.alpha[n] * fg_prototypes[n][way] for n in range(len(supp_fts))]
                fg_prototypes = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0) / torch.sum(self.alpha)
                supp_prototypes_ = [self.alpha[n] * supp_prototypes[n][way] for n in range(len(supp_fts))]
                supp_prototypes_ = torch.sum(torch.stack(supp_prototypes_, dim=0), dim=0) / torch.sum(self.alpha)

                # Compute the MSE loss
                loss_sim += self.criterion_MSE(fg_prototypes, supp_prototypes_)

        return loss_sim


    def get_pts(self, fts, mask):
        features_trans = fts.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1
        features_trans = features_trans[indx]
        if len(features_trans) > 100:
            features_trans = features_trans[random.sample(range(len(features_trans)), 100)]  # 这是全是背景的情况，随机选择100个背景特征点
        return features_trans


    def scan(self, fts, mask, kernel_size=(16, 16), stride=8, flag=1):

        b, c, height, width = fts.size()
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)

        kernel_h, kernel_w = kernel_size
        n_fg = mask.sum()
        n_bg = height * width - n_fg
        fg_pts = []
        # bg_pts = []

        fg_pts.append(torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (n_fg + 1e-5)) #前景原型
        patch_fts = F.unfold(fts, kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1, c, kernel_h, kernel_w)
        patch_mask = F.unfold(mask.unsqueeze(1), kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1, kernel_h, kernel_w).unsqueeze(2)

        bg_patch_mask = 1 - patch_mask

        if flag == 1:
            fg_indx = torch.sum(patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h*kernel_w*self.rate, n_fg/100) #torch.Size([961])
            fg_protos = torch.sum(patch_fts[:, fg_indx, :, :, :] * patch_mask[:, fg_indx, :, :, :], dim=(-2, -1)) \
                        / (patch_mask[:, fg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)

            if fg_protos.numel() == 0:
                fg_protos = self.get_pts(fts, mask)
            else:
                fg_protos = fg_protos.squeeze(0)
                if len(fg_protos) > 150:
                    fg_protos = fg_protos[random.sample(range(len(fg_protos)), 150)]
            fg_pts.append(fg_protos)

        fg_pts = torch.cat(fg_pts, dim=0)

        return fg_pts.unsqueeze(0)

    def generate_prior(self, query_feat_high, final_supp_list, mask_list, fts_size):

        bsize, ch_sz, sp_sz, _ = query_feat_high.size()[:]
        corr_query_mask_list = []
        cosine_eps = 1e-7

        for i, tmp_supp_feat in enumerate(final_supp_list):

            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat = tmp_supp_feat
            q = query_feat_high.flatten(2).transpose(-2, -1)
            s = tmp_supp_feat.flatten(2).transpose(-2, -1)

            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)

            similarity = similarity.permute(0, 2, 1)

            similarity = F.softmax(similarity, dim=-1)
            similarity = torch.bmm(similarity, tmp_mask.flatten(2).transpose(-2, -1)).squeeze(-1)


            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)

            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=fts_size, mode='bilinear', align_corners=True)

            corr_query_mask_list.append(corr_query)

        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (corr_query_mask).mean(1, True)

        return corr_query_mask





