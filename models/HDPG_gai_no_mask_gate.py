import torch
import torch.nn as nn
import torch.nn.functional as F


class HDPGModule(nn.Module):


    def __init__(self, in_channels_list, out_channels=256, reduction=16):

        super(HDPGModule, self).__init__()

        self.num_levels = len(in_channels_list)
        self.high_level_idx = self.num_levels - 1

        self.align_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_list[i], out_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            ) for i in range(self.high_level_idx)
        ])

        self.low_fusion = nn.Sequential(
            nn.Conv2d(out_channels * (self.num_levels - 1), out_channels, kernel_size=1, bias=False), #512*4,512；512*3,512
            nn.ReLU(inplace=True)
        )

        self.channel_gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_list):

        target_size = feat_list[self.high_level_idx].shape[2:] #[32,32];#[64,64]

        aligned_low_feats = []
        for i in range(self.high_level_idx):

            f_resized = F.interpolate(feat_list[i], size=target_size, mode='bilinear', align_corners=True)
            aligned_low_feats.append(self.align_layers[i](f_resized))


        f_low = self.low_fusion(torch.cat(aligned_low_feats, dim=1))

        z_avg = F.adaptive_avg_pool2d(f_low, 1)
        z_max = F.adaptive_max_pool2d(f_low, 1)
        a_s = self.channel_gate(z_avg + z_max)  # [B, C, 1, 1]
        f_low_channel_refined = f_low * a_s

        s_g  = self.spatial_gate(f_low_channel_refined)
        f_low_refined = f_low_channel_refined * s_g

        f_high = feat_list[self.high_level_idx]

        f_fus = f_high + self.gamma * f_low_refined

        return f_fus, self.gamma


if __name__ == "__main__":

    # f1 = torch.randn(1, 64, 64, 64)
    f2 = torch.randn(1, 256, 64, 64)
    f3 = torch.randn(1, 512, 64, 64)
    f4 = torch.randn(1, 1024, 64, 64)
    f5 = torch.randn(1, 2048, 32, 32)  #

    feats = [f2, f3, f4, f5]
    # feats = [f1, f2, f3, f4]

    mask = (torch.randn(1, 1, 256, 256) > 0).float()

    hdpg = HDPGModule(in_channels_list=[ 256, 512, 1024, 2048], out_channels=2048)
    # hdpg = HDPGModule(in_channels_list=[64, 256, 512, 1024], out_channels=1024)

    prototype = hdpg(feats, mask)

    print(f"Final Refine Shape: {prototype.shape}")
