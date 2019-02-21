# Adapted from https://github.com/fregu856/deeplabv3

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from models._deeplabv3_resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from models.deeplabv3_aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, pretrained, num_classes, **kwargs):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes

        self.resnet = ResNet18_OS8(pretrained) # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

def deeplabv3(output_channels, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DeepLabV3(pretrained, output_channels, **kwargs)
    # if pretrained:
    #     try:
    #         model.load_state_dict(torch.load(os.path.join(os.getcwd(), "pretrained_models/model_13_2_2_2_epoch_580.pth")))
    #         # reinitialize weights in the last layer
    #         model.aspp.conv_1x1_4 = nn.Conv2d(256, output_channels, kernel_size=1)
    #
    #     except Exception as exp:
    #         logging.warning(exp)

    return model