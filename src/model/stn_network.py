# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/20/2020 11:31 AM
# @Author : Zhuoyi Huang
# @File : stn_network.py

import torch.nn as nn
import torch
import torch.nn.functional as F


class StnNetwork(nn.Module):
    def __init__(self):
        super(StnNetwork, self).__init__()  # call the construction function of the parent class
        # (Spatial transformer localization-network)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.localization2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # 3 * 2  (affine matrix) regress
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 60, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc2 = nn.Sequential(
            nn.Linear(10 * 60 * 60, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # use (identity transformation) initialize  (weights) /  (bias)
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

        self.fc_loc2[2].weight.data.fill_(0)
        self.fc_loc2[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    #  (Spatial transformer network forward function)
    def stn(self, x, y):
        xs = self.localization(x)
        ys = self.localization2(y)
        xs = xs.view(-1, 10 * 60 * 60)
        ys = ys.view(-1, 10 * 60 * 60)

        theta_1 = self.fc_loc(xs)
        theta_2 = self.fc_loc2(ys)
        theta_1 = theta_1.view(-1, 2, 3)
        theta_2 = theta_2.view(-1, 2, 3)
        grid_1 = F.affine_grid(theta_1, x.size())
        grid_2 = F.affine_grid(theta_2, y.size())

        # x_p = x.detach().numpy()
        # print("x_size", x.size())
        # print("x_origin", x_p, file=f)

        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        x = F.grid_sample(x, grid_1)
        y = F.grid_sample(y, grid_2)

        # x_t_p = x.detach().numpy()
        # print("x_transformed", x_t_p, file=f)

        # x_r_p = x.detach().numpy()
        # print("x_regularized", x_r_p, file=f)
        ones = ones * 0.9

        output = x + y - ones
        output = torch.clamp(output, -1.0, 1.0)
        # output = output - ones
        # output = torch.clamp(output, -1.0, 1.0)
        # output = torch.mul(x, y)
        # print("torch.mul", output)
        # output_p = output.detach().numpy()
        # print("output = x*y", output_p, file=f)
        return output, x, y

    def forward(self, x, y):
        # 转换输入
        output, x, y = self.stn(x, y)
        return output, x, y

def build_model():
    model = StnNetwork()
