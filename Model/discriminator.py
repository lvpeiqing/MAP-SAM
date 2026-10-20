
import torch
import torch.nn as nn
class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=2, out_conv_channels=1):
        super(Discriminator, self).__init__()
        ambiguous_channels = 16
        entmap_channels = 16
        logits_channels = 16
        embedding_channels = 64

        self.out_conv_channels = out_conv_channels

        self.ambiguous_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ambiguous_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ambiguous_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ambiguous_channels, out_channels=ambiguous_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ambiguous_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.entmap_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=entmap_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(entmap_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=entmap_channels, out_channels=entmap_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(entmap_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.logits_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=logits_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(logits_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=logits_channels, out_channels=logits_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(logits_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=embedding_channels, out_channels=embedding_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=embedding_channels, out_channels=out_conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def get_entropy_map(self, p):
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map

    def forward(self, pred_UNet, pred_YNet, pred_UNet_soft,  pred_VNet_soft, entmap1, entmap2, thr=0.5):

        #大于阈值 thr 设为1，否则0
        pred_UNet_bool = torch.where(pred_UNet_soft > thr, torch.tensor(1).to(pred_UNet_soft.device), torch.tensor(0).to(pred_UNet_soft.device))
        pred_YNet_bool = torch.where(pred_VNet_soft > thr, torch.tensor(1).to(pred_UNet_soft.device), torch.tensor(0).to(pred_UNet_soft.device))

        #异或操作，取不重叠的区域。即不重叠的区域为1，重叠的为0
        ambiguous_area = torch.bitwise_xor(pred_UNet_bool, pred_YNet_bool).to(dtype=torch.float32) #torch.Size([24, 4, 256, 256])
        #两个网络的不确定性熵图进行拼接
        uncertainty_area = torch.cat((entmap1, entmap2), dim=1) #torch.Size([24, 2, 256, 256])
        #两个网络的logit预测结果进行拼接
        pred_logits = torch.cat((pred_UNet, pred_YNet), dim=1) #torch.Size([24, 8, 256, 256])

        ##########################通过卷积分支提取特征###########################
        #1 - ambiguous_area（反转关注点，可能是为了强调一致区域，或者单纯作为特征输入）。
        #1 - uncertainty_area（反转熵图，值越大表示越确定）。
        ambiguous_info = self.ambiguous_conv(1 - ambiguous_area)
        uncertainty_info = self.entmap_conv(1 - uncertainty_area)
        pred_info = self.logits_conv(pred_logits)
        # print(ambiguous_info.shape,uncertainty_info.shape,pred_info.shape)
        #torch.Size([24, 16, 256, 256]) torch.Size([24, 16, 256, 256]) torch.Size([24, 32, 256, 256])

        #特征融合
        x = torch.cat((ambiguous_info, uncertainty_info, pred_info), dim=1)  #torch.Size([24, 64, 256, 256])

        x = self.final_conv(x) #torch.Size([24, 4, 256, 256])

        return x