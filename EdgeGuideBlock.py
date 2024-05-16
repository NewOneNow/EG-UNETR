import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

class fpnblock(nn.Module):
    def __init__(self):
        super(fpnblock,self).__init__()

        self.conv2=nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up23 = nn.ConvTranspose2d(64,64,2,2)
        self.up34 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.up45 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.upforcat2 = nn.Upsample(size=(512,512),mode='bilinear')
        self.upforcat3 = nn.Upsample(size=(512,512),mode='bilinear')
        self.upforcat4 = nn.Upsample(size=(512,512),mode='bilinear')
        self.cv = nn.Conv2d(256,64,1)
    def forward(self,x2,x3,x4,x5):
        x2=self.conv2(x2)
        x3=self.conv3(x3)
        x4=self.conv4(x4)
        x5=self.conv5(x5)

        up23=x3+self.up23(x2)
        up34=x4+self.up34(up23)
        up45=x5+self.up45(up34)

        merge=torch.cat((up45,self.upforcat2(x2),
                         self.upforcat3(x3),
                         self.upforcat4(x4)
                         ),dim=1)
        return self.cv(merge)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return x*out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x*self.sigmoid(out)

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv: # 跨连path是否使用1*1conv
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True) # inplace方式节省内存，训练更快
        self.se = SELayer(num_channels)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.se(y)
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x # 残差concate
        return F.relu(y)

class EdgeWeightAttention(nn.Module):
    def __init__(self, in_planes):
        super(EdgeWeightAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(edge))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(edge))))
        # 结果相加
        out = avg_out + max_out
        return x*self.sigmoid(out)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

device = torch.device('cpu')

class EdgeGuideAttention(nn.Module):
    def __init__(self):
        super(EdgeGuideAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edgeweight):
        # map尺寸不变，缩减通道
        avgout = torch.mean(edgeweight, dim=1, keepdim=True)
        maxout, _ = torch.max(edgeweight, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(F.relu(self.conv2d(out)))
        return x*out

class edgeguide(nn.Module):
    def __init__(self, ch):
        super(edgeguide, self).__init__()
        self.convdown = nn.Conv2d(64,2,1)
        self.sig = nn.Sigmoid()
        self.fpn = fpnblock()
        self.ewa = EdgeWeightAttention(ch)
        self.se = SELayer(ch)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        self.conv = nn.Conv2d(64, ch, 1,1)
        self.res_se = Residual(64, 64)
        self.edgeattention = EdgeGuideAttention()
    def forward(self, f2, f3, f4, f5, edge):
        edge=edge.to(device)
        merge = self.fpn(f2, f3, f4, f5)
        camerge = self.ca(merge).to(device)
        downmerge = F.relu(self.convdown(camerge))
        multip=torch.mul(downmerge,edge)
        add = self.sa(multip)
        lse = self.ewa(add,edge)

        res_se = self.res_se(merge)

        final = self.edgeattention(res_se,lse)
        final_predict_result = self.conv(final)

        return final_predict_result
