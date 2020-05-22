# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/22/2020 9:04 AM
# @Author : Zhuoyi Huang
# @File : loss.py

import torch
import torchvision
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class VGG(nn.Module):
#     def __init__(self, features):
#         super(VGG, self).__init__()
#         self.features = features
#         self.layer_name_mapping = {
#             '3': "relu1_2",
#             '8': "relu2_2",
#             '15': "relu3_3",
#             '22': "relu4_3"
#         }
#         for p in self.parameters():
#             p.requires_grad = False
#
#     def forward(self, x):
#         outs = []
#         for name, module in self.features._modules.items():
#             x = module(x)
#             if name in self.layer_name_mapping:
#                 outs.append(x)
#         return outs

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        # if self.resize:
        #     input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        #     target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss