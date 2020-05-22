# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/22/2020 5:32 AM
# @Author : Zhuoyi Huang
# @Site : 
# @File : test.py
# @Software:
import os
item, item2, styles, label = [], [], [], []
item += [1, 2, 3]
item2 += [[2, 3], [4, 5], [6, 7]]
styles += ['Regular'] * 3
label += [1] * 3

fname = "../../data/interFiles/FontImages/Regular/comp_256/lid1_04ebf_04ebb_04e59.png"

basename = os.path.basename(fname).split(".")[0]
words = list(basename.split("_"))

print(words)
