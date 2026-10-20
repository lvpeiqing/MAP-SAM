"""
Backbones supported by torchvison.
"""
import torch
import torch.nn as nn
import torchvision
from .HDPG_gai_no_mask_gate import HDPGModule


class Res101Encoder(nn.Module):
    """
    Resnet101 backbone from deeplabv3
    modify the 'downsample' component in layer2 and/or layer3 and/or layer4 as the vanilla Resnet
    """

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights='resnet101'):
        super().__init__()
        # using pretrained model's weights
        if pretrained_weights == 'deeplabv3':
            self.pretrained_weights = torch.load(
                "/data3/lpq/few-shot/MAPSAM/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth", map_location='cpu')
        elif pretrained_weights == 'resnet101':
            self.pretrained_weights = torch.load("/data3/xxx/few-shot/MAPSAM/checkpoints/resnet101-63fe2227.pth",
                                                 map_location='cpu')
        else:
            self.pretrained_weights = pretrained_weights

        _model = torchvision.models.resnet.resnet101(pretrained=False,
                                                     replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.ModuleDict()
        for dic, m in _model.named_children():
            self.backbone[dic] = m

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)
        self.hdpg1 = HDPGModule(in_channels_list=[256, 512, 1024], out_channels=1024)
        self.hdpg2 = HDPGModule(in_channels_list=[256, 512, 1024, 2048], out_channels=2048)

        self._init_weights()

    def forward(self, x):
        features = dict()
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)

        x_0 = self.backbone["maxpool"](x)
        x_1 = self.backbone["layer1"](x_0)
        x_2 = self.backbone["layer2"](x_1)
        x_3 = self.backbone["layer3"](x_2)
        x_3_query = x_3[1:2]

        x_4 = self.backbone["layer4"](x_3)
        x_4_query = x_4[1:2]

        x_1_supp = x_1[0:1]
        x_2_supp = x_2[0:1]
        x_3_supp = x_3[0:1]
        x_4_supp = x_4[0:1]

        feats_1 = [x_1_supp,x_2_supp,x_3_supp]
        feats_2 = [x_1_supp,x_2_supp,x_3_supp,x_4_supp]

        x_3_supp_refine,gamma_1 = self.hdpg1(feats_1)
        x_4_supp_refine,gamma_2 = self.hdpg2(feats_2)

        x_3_supp_refine_query = torch.cat([x_3_supp_refine, x_3_query], dim=0)
        x_4_supp_refine_query = torch.cat([x_4_supp_refine, x_4_query], dim=0)

        features['down2'] = self.reduce1(x_3_supp_refine_query)
        features['down3'] = self.reduce2(x_4_supp_refine_query)

        # feature map -> avgpool -> fc -> single value
        t = self.backbone["avgpool"](x_4_supp_refine_query)  # (2, 2048, 1, 1)
        t = torch.flatten(t, 1)  # (2, 2048)
        t = self.backbone["fc"](t)  # (2, 1000)
        t = self.reduce1d(t)  # (2, 1)

        return (features, t, gamma_1, gamma_2)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.pretrained_weights is not None:
            keys = list(self.pretrained_weights.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(len(keys)):
                if keys[i] in new_keys:
                    new_dic[keys[i]] = self.pretrained_weights[keys[i]]

            self.load_state_dict(new_dic)

