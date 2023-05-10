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

#start

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

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
                 block = BottleNeck,
                 layers=[3,4,6,3],
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
        #start
        nb_filter = [16,32,64, 128, 256]
        self.in_channels = nb_filter[0]
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(BottleNeck, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(BottleNeck, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(BottleNeck, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(BottleNeck, nb_filter[4], layers[3], 1)

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1] * BottleNeck.expansion, nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock((nb_filter[1] + nb_filter[2]) * BottleNeck.expansion, nb_filter[1],
                                nb_filter[1] * BottleNeck.expansion)
        self.conv2_1 = VGGBlock((nb_filter[2] + nb_filter[3]) * BottleNeck.expansion, nb_filter[2],
                                nb_filter[2] * BottleNeck.expansion)
        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * BottleNeck.expansion, nb_filter[3],
                                nb_filter[3] * BottleNeck.expansion)

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1] * BottleNeck.expansion, nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock((nb_filter[1] * 2 + nb_filter[2]) * BottleNeck.expansion, nb_filter[1],
                                nb_filter[1] * BottleNeck.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] * 2 + nb_filter[3]) * BottleNeck.expansion, nb_filter[2],
                                nb_filter[2] * BottleNeck.expansion)

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1] * BottleNeck.expansion, nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock((nb_filter[1] * 3 + nb_filter[2]) * BottleNeck.expansion, nb_filter[1],
                                nb_filter[1] * BottleNeck.expansion)

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1] * BottleNeck.expansion, nb_filter[0], nb_filter[0])


        self.final = nn.Conv2d(nb_filter[0], 64, kernel_size=1)

    def _make_layer(self, block, middle_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, middle_channels, stride))
            self.in_channels = middle_channels * block.expansion
        return nn.Sequential(*layers)
        #end

    def forward(self, x, with_additional=False):
        [global_contexual, regional_distribution, region_coeff] = self.transformer(x)
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
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
        #
        # for i, block in enumerate(self.up_blocks, 1):
        #     key = f"layer_{TransMUNet.DEPTH - 1 - i}"
        #     x = block(x, pre_pools[key])
        #start

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print("s333", np.shape(x0_4))
        x = self.final(x0_4)
        # print("sss", np.shape(x))
        #end

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


# #
# F1 score (F-measure) or DSC: 0.8752448415873152
# Accuracy: 0.9525223159790039
# Specificity: 0.9655539140585866
# Sensitivity: 0.8931122237464351
# val_mode
# 4it [00:09,  2.34s/it]

#