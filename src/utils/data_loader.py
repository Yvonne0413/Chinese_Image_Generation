# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/22/2020 3:47 AM
# @Author : Zhuoyi Huang
# @File : data_loader.py

import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
from pathlib import Path
from itertools import chain
from PIL import Image
import os
import random
import numpy as np
import torch
from munch import Munch

def listdir(dname):
    """
    get all image file paths inside directory dname, can be subfile
    :param dname:
    :return: a list of all image file paths
    """
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class TrainDataset(data.Dataset):
    def __init__(self, root, style_name, augment=True, transform=None):
        super(TrainDataset, self).__init__()
        # labels is the int version of lid
        self.style_name = style_name
        self.augment = augment
        self.transform = transform

        if root:
            # root example ../../data/interFiles/FontImages:
            self.comp_dir = os.path.join(root, style_name, "comp_256/")
            self.base_img_dir = os.path.join(root, style_name, "origin_256/")

        self.samples, self.labels = self.make_dataset()  # label means lid number, lid0 = 0 or lid1 = 1...

    def make_dataset(self):
        domains = os.listdir(self.comp_dir)
        trg_fnames, src_fname_lists, basename_list, labels = [], [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(self.comp_dir, domain)
            cls_fnames = listdir(class_dir)  # cls_fnames is the list of all img paths in this domain
            trg_fname, src_fname_list, basenames = self.process(cls_fnames)  # return two list, a list for trg img and
            # a list for src_img_list in this domain
            trg_fnames += trg_fname
            src_fname_lists += src_fname_list
            basename_list += basenames
            # print("idx", idx)
            labels += [int(idx)] * len(cls_fnames)  # give all items idx and fonts style as label
        return list(zip(trg_fnames, src_fname_lists, basename_list)), labels

    def process(self, cls_fnames):
        trg_fnames, src_fname_lists, basenames = [], [], []
        for fname in cls_fnames:
            basename = os.path.basename(fname).split(".")[0]
            words = list(basename.split("_"))
            length = len(words)
            trg_uid = words[1][1:]
            trg_fname = self.base_img_dir + str(trg_uid) + ".png"
            src_fname = []
            for i in range(2, length):
                src_uid = words[i][1:]
                fname = self.base_img_dir + str(src_uid) + '.png'
                src_fname.append(fname)
            trg_fnames.append(trg_fname)
            src_fname_lists.append(src_fname)
            basenames.append(basename)
        return trg_fnames, src_fname_lists, basenames  # example [path1, path2] [[path3, path4], [path5, path6]]

    def __getitem__(self, index):  # get specific image at index
        trg_fname = self.samples[index][0]
        src_fname_list = self.samples[index][1]
        style_name = self.samples[index][2]
        label = self.labels[index]
        src_len = len(src_fname_list)

        trg_img = Image.open(trg_fname).convert('1')
        src_img_list = []
        for i in range(src_len):
            img = Image.open(src_fname_list[i]).convert('1')
            src_img_list.append(img)

        if self.transform is not None:

            # augment the image by:
            # 1) enlarge the image
            # 2) random crop the image back to its original size
            # NOTE: image A and B needs to be in sync as how much
            # to be shifted
            if self.augment:
                w, h = trg_img.size
                multiplier = random.uniform(1.00, 1.20)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                w_offset = random.randint(0, max(0, nh - h - 1))
                h_offset = random.randint(0, max(0, nh - h - 1))
                trg_img = trg_img.resize((nw, nh), Image.BICUBIC)
                trg_img = transforms.ToTensor()(trg_img)
                trg_img = trg_img[:, h_offset: h_offset + h, w_offset: w_offset + h]
                trg_img = self.transform(trg_img)
                for i in range(len(src_img_list)):
                    img = src_img_list[i].resize((nw, nh), Image.BICUBIC)
                    img = transforms.ToTensor()(img)
                    img = img[:, h_offset: h_offset + h, w_offset: w_offset + h]
                    img = self.transform(img)
                    src_img_list[i] = img

        return trg_img, src_img_list, style_name, label

    def __len__(self):
        return len(self.labels)


class TestDataset(data.Dataset):
    def __init__(self, root, style_name, augment=True, transform=None):
        super(TestDataset, self).__init__()
        # labels is the int version of lid
        self.style_name = style_name
        self.augment = augment
        self.transform = transform

        if root:
            # root example ../../data/interFiles/FontImages:
            self.comp_dir = os.path.join(root, style_name, "comp_256/")
            self.base_img_dir = os.path.join(root, style_name, "origin_256/")

        self.samples, self.labels = self.make_dataset()  # label means lid number, lid0 = 0 or lid1 = 1...

    def make_dataset(self):
        domains = os.listdir(self.comp_dir)
        trg_fnames, src_fname_lists, basename_list, labels = [], [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(self.comp_dir, domain)
            cls_fnames = listdir(class_dir)
            # cls_fnames is the list of all img paths in this domain
            trg_fname, src_fname_list, basenames = self.process(cls_fnames)
            # return two list, a list for trg img and
            # a list for src_img_list in this domain
            trg_fnames += trg_fname
            src_fname_lists += src_fname_list
            basename_list += basenames
            labels += [int(idx)] * len(cls_fnames)
            # give all items idx and fonts style as label
        return list(zip(trg_fnames, src_fname_lists, basename_list)), labels

    def process(self, cls_fnames):
        trg_fnames, src_fname_lists, basenames = [], [], []
        for fname in cls_fnames:
            basename = os.path.basename(fname).split(".")[0]
            words = list(basename.split("_"))
            length = len(words)
            trg_uid = words[1][1:]
            trg_fname = self.base_img_dir + str(trg_uid) + ".png"
            src_fname = []
            for i in range(2, length):
                src_uid = words[i][1:]
                fname = self.base_img_dir + str(src_uid) + '.png'
                src_fname.append(fname)
            trg_fnames.append(trg_fname)
            src_fname_lists.append(src_fname)
            basenames.append(basename)
        return trg_fnames, src_fname_lists, basenames
        # example [path1, path2], [[path3, path4], [path5, path6]], [str1, str2]

    def __getitem__(self, index):  # get specific image at index
        trg_fname = self.samples[index][0]
        src_fname_list = self.samples[index][1]
        style_name = self.samples[index][2]
        label = self.labels[index]
        src_len = len(src_fname_list)

        trg_img = Image.open(trg_fname).convert('1')
        src_img_list = []
        for i in range(src_len):
            img = Image.open(src_fname_list[i]).convert('1')
            src_img_list.append(img)

        if self.transform is not None:
            trg_img = self.transform(trg_img)
            for i in range(src_len):
                src_img = src_img_list[i]
                src_img_list[i] = self.transform(src_img)

        return trg_img, src_img_list, style_name, label

    def __len__(self):
        return len(self.labels)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)  # [counts for label 0, counts for label 1,....]
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))  # for balanced sampling


class InputFetcher:
    def __init__(self, loader, latent_dim=16):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_inputs(self):
        try:
            t_img, s_list, s_name, label = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            t_img, s_list, s_name, label = next(self.iter)
        return t_img, s_list, s_name, label

    def __next__(self):
        t_img, s_list, s_name, label = self._fetch_inputs()
        inputs = Munch(t_img=t_img, s_list=s_list, s_name=s_name, label=label)

        return Munch({k: v for k, v in inputs.items()})




