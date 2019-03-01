"""
Convolutional Auto Encoder with 3 conv layers and a fully connected classification layer
"""

import torch.nn as nn
import torch


class BabyUnet(nn.Module):
    """
    Simple convolutional auto-encoder neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    """
    def __init__(self, input_channels=3, output_channels=3, num_filter=24, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        output_channels : int
            Number of classes
        """

        super(BabyUnet, self).__init__()
        self.in_dim = input_channels
        self.out_dim = output_channels
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        self.down_1 = conv_block_2(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool()

        self.bridge = conv_block_2(self.num_filter * 2, self.num_filter * 4, act_fn)

        self.trans_1 = conv_trans_block(self.num_filter * 4, self.num_filter * 2, act_fn)
        self.up_1 = conv_block_2(self.num_filter * 4, self.num_filter * 2, act_fn)
        self.trans_2 = conv_trans_block(self.num_filter * 2, self.num_filter, act_fn)
        self.up_2 = conv_block_2(self.num_filter * 2, self.num_filter, act_fn)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        bridge = self.bridge(pool_2)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_2], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_1], dim=1)
        up_2 = self.up_2(concat_2)

        out = self.out(up_2)
        return out


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model


def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model
