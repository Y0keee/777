import torch
import torch.nn as nn
import torchvision

resnet = torchvision.models.resnet.resnet50(pretrained=True)
from .transformer import ViT
import cv2
import numpy as np


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        return y

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class TransMUNet(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2,
                 patch_size: int = 16,
                 emb_size: int = 512,
                 img_size: int = 256,
                 n_channels=3,
                 depth: int = 4,
                 n_regions: int = (256 // 16) ** 2,
                 output_ch: int = 1,
                 bilinear=True):
        super().__init__()
        self.n_classes = n_classes
        self.transformer = ViT(in_channels=n_channels,
                               patch_size=patch_size,
                               emb_size=emb_size,
                               img_size=img_size,
                               depth=depth,
                               n_regions=n_regions)
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(128, n_classes, kernel_size=1, stride=1)

        self.boundary = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
                                      nn.Sigmoid())

        self.se = SE_Block(c=64)
        #art
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # backbone中下采样的各个级别层
        self.conv0_0 = conv_block_nested(3, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        # x^(0,1),x^(1,1),x^(2,1),x^(3,1)层
        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        ##x^(0,2),x^(1,2),x^(2,2)层
        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        # x^(0,3),x^(1,3)层
        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        # x^(0,4)层
        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], 64, kernel_size=1)
        #end

    def forward(self, x, with_additional=False):
        [global_contexual, regional_distribution, region_coeff] = self.transformer(x)
        pre_pools = dict()
        # pre_pools[f"layer_0"] = x
        # x = self.input_block(x)
        # pre_pools[f"layer_1"] = x
        # x = self.input_pool(x)
        #
        # for i, block in enumerate(self.down_blocks, 2):
        #     x = block(x)
        #     if i == (TransMUNet.DEPTH - 1):
        #         continue
        #     pre_pools[f"layer_{i}"] = x
        #
        # x = self.bridge(x)
        # #a = self.bridge(x)
        #
        # for i, block in enumerate(self.up_blocks, 1):
        #     key = f"layer_{TransMUNet.DEPTH - 1 - i}"
        #     x = block(x, pre_pools[key])
        #
        # # #start
        # #
        # # x0_0 = pre_pools[1]
        # #
        # # x1_0 = pre_pools[2]
        # # x2_0 = pre_pools[3]
        # # x3_0 = pre_pools[4]
        # # x4_0 = a
        # # #end
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))
        #print("x0_4", np.shape(x0_4))
        x = self.final(x0_4)



        B_out = self.boundary(x)
        B = B_out.repeat_interleave(int(x.shape[1]), dim=1)
        x = self.se(x)

        x = x + B
        att = regional_distribution.repeat_interleave(int(x.shape[1]), dim=1)
        x = x * att
        x = torch.cat((x, global_contexual), dim=1)
        x = self.out(x)
        del pre_pools
        x = torch.sigmoid(x)
        if with_additional:
            return x, B_out, region_coeff
        else:
            return x


#17isic1

# F1 score (F-measure) or DSC: 0.8782599866747688
# 0it [00:00, ?it/s][[20815727   682942]
#  [  528098  4187633]]
# Accuracy: 0.953802490234375
# Specificity: 0.9682332892329288
# Sensitivity: 0.8880135444536594
# val_mode