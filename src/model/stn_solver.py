# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/22/2020 7:23 AM
# @Author : Zhuoyi Huang
# @File : stn_solver.py

import argparse
import torch.nn as nn
import torch
import model.stn_network as stn

parser = argparse.ArgumentParser(description='STN network')
parser.add_argument('--train_start_iter', dest='train_start_iter', default=0,
                    help='the start iter of training network')
parser.add_argument('--train_end_iter', dest='train_end_iter', default=100,
                    help='the start iter of training network')
parser.add_argument('--config_lid', dest='config_lid', default=1,
                    help='config_lid, range from 0 to 38')
parser.add_argument('--batch_size', dest='batch_size', default=4,
                    help='batch_size')
parser.add_argument('--mode', dest='mode', type=str, default='train',
                    choices=['train', 'test'],
                    help='different mode for the usage of the network.')
parser.add_argument('--data_root', dest='data_root', default="../../data/interFiles/FontImages/",
                    help='input data directory default "../../data/interFiles/FontImages/"')
parser.add_argument('--style_name', dest='style_name', default="Regular",
                    help='fonts style')
parser.add_argument('--output_backup', dest='output_backup', default="../../data/output.bak",
                    help='output backup directory')
args = parser.parse_args()


class StnSolver(nn.Module):
    def __init__(self, args, style_name):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = stn.build_model
