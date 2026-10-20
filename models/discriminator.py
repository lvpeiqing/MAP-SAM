import torch
import torch.nn as nn


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=2, out_conv_channels=2):
        super(Discriminator, self).__init__()

        ambiguous_channels = 16
        entmap_channels = 16
        logits_channels = 16
        embedding_channels = 64
        self.device = torch.device('cuda')

        self.out_conv_channels = out_conv_channels

        self.ambiguous_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=ambiguous_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ambiguous_channels, out_channels=ambiguous_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.entmap_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=entmap_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=entmap_channels, out_channels=entmap_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.logits_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=logits_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=logits_channels, out_channels=logits_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=embedding_channels, out_channels=embedding_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embedding_channels, out_channels=out_conv_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def get_entropy_map(self, p):
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map

    def forward(self, query_pred, R_gradient):

        pred_logits = torch.logit(query_pred[:, 1:2, ...], eps=1e-5)
        pred_info = self.logits_conv(pred_logits)

        uncertainty_area = self.get_entropy_map(query_pred)
        uncertainty_info = self.entmap_conv(uncertainty_area)

        R_gradient = torch.from_numpy(R_gradient).unsqueeze(0).unsqueeze(1).cuda()
        morphological_gradient_info = self.ambiguous_conv(R_gradient.float())

        x = torch.cat((morphological_gradient_info, uncertainty_info, pred_info), dim=1)
        x = self.final_conv(x)

        return x,uncertainty_area,R_gradient