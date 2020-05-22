# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/22/2020 7:23 AM
# @Author : Zhuoyi Huang
# @File : stn_solver.py

import os
import time
import argparse
import imageio
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch
import torchvision
import model.stn_network as stn
from torch.utils.data import DataLoader

from utils.data_loader import TrainDataset, TestDataset, InputFetcher
from model.loss import VGGPerceptualLoss

parser = argparse.ArgumentParser(description='STN network')
parser.add_argument('--start_iter', dest='start_iter', default=0,
                    help='the start iter of training network')
parser.add_argument('--end_iter', dest='end_iter', default=100,
                    help='the start iter of training network')
parser.add_argument('--batch_size', dest='batch_size', default=4,
                    help='batch_size')
parser.add_argument('--lr', dest='lr', default=0.005,
                    help='batch_size')
parser.add_argument('--img_size', dest='img_size', default=256,
                    help='image size to be processed')

parser.add_argument('--config_lid', dest='config_lid', default=1,
                    help='config_lid, range from 0 to 38')
parser.add_argument('--style_name', dest='style_name', default="Regular",
                    help='fonts style')

parser.add_argument('--mode', dest='mode', type=str, default='train',
                    choices=['train', 'test'],
                    help='different mode for the usage of the network.')

parser.add_argument('--data_root', dest='data_root', default="../../data/interFiles/FontImages/",
                    help='input data directory default "../../data/interFiles/FontImages/"')
parser.add_argument('--output_backup', dest='output_backup', default="../../data/output.bak",
                    help='output backup directory')
parser.add_argument('--ckp_dir', dest='ckp_dir', default="../../experiment/ckp/stn/",
                    help='ckp model save directory')
parser.add_argument('--results_dir', dest='results_dir', default="../../experiment/results/",
                    help='eval save directory')
parser.add_argument('--print_every', dest='print_every', default=10,
                    help='print_every epoch')
args = parser.parse_args()


class StnSolver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.train_net = stn.build_model()
        self.vgg16 = VGGPerceptualLoss()
        self.criterion = nn.MSELoss()

        transform = transforms.Compose([
            # transforms.Resize([self.args.img_size, self.args.img_size]),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,), (0.5,))
        ])

        if not os.path.exists(self.args.ckp_dir):
            os.makedirs(self.args.ckp_dir)

        if not os.path.exists(self.args.results_dir):
            os.makedirs(self.args.results_dir)

        if self.args.data_root:
            train_dataset = TrainDataset(self.args.data_root, self.args.style_name, transform=transform)
            val_dataset = TestDataset(self.args.data_root, self.args.style_name, transform=transform)
            self.data_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=True)

        if args.mode == 'train':
            self.optim = optim.SGD(self.train_net.parameters(), self.args.lr)

        print(self.train_net)

    def restore_checkpoint(self, resume_iter):
        print('Loading the trained models from step {}...'.format(resume_iter))
        resume_path = os.path.join(self.args.ckp_dir, self.args.style_name, self.args.config_lid,
                                   '{}-Model.ckpt'.format(resume_iter))
        self.train_net.load_state_dict(torch.load(resume_path, map_location=lambda storage, loc: storage))

    def save_checkpoint(self, cur_iter):
        print('Saving the trained models from step {}...'.format(cur_iter))
        save_dir = os.path.join(self.args.ckp_dir, self.args.style_name, self.args.config_lid)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, '{}-Model.ckpt'.format(cur_iter+1))
        torch.save(self.train_net.state_dict(), save_path)

    @staticmethod
    def print_network(network, name):
        num_params = 0
        for p in network.parameters():
            num_params += p.numel()
        print(network)
        print("Number of parameters of %s: %i" % (name, num_params))

    @staticmethod
    def convert_image_np(inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        inp = np.where(inp < 0, -1, 1)
        return inp

    def draw_inter(self, comp_1, comp_2, target, output, output_1, output_2, basename, label, epoch):
        input_tensor1 = comp_1.cpu().data
        input_tensor2 = comp_2.cpu().data
        target_tensor = target.cpu().data
        output_tensor = output.cpu().data
        output_1 = output_1.cpu().data
        output_2 = output_2.cpu().data

        in_grid1 = self.convert_image_np(torchvision.utils.make_grid(input_tensor1))
        in_grid2 = self.convert_image_np(torchvision.utils.make_grid(input_tensor2))
        out_grid1 = self.convert_image_np(torchvision.utils.make_grid(output_1))
        out_grid2 = self.convert_image_np(torchvision.utils.make_grid(output_2))
        output_grid = self.convert_image_np(torchvision.utils.make_grid(output_tensor))
        target_grid = self.convert_image_np(torchvision.utils.make_grid(target_tensor))

        result = np.concatenate([in_grid1, in_grid2, out_grid1, out_grid2, output_grid, target_grid], axis=1)
        result_2 = np.concatenate([target_grid, output_grid], axis=1)
        if self.args.mode == "train":
            save_dir = os.path.join(self.args.results_dir, self.args.style_name, self.args.config_lid, str(epoch))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            words = list(basename.split("_"))  # basename = lidX_target_uid_comp1_uid_comp2_uid
            lid = words[0]
            target_uid = words[1]
            comp_1_uid = words[2]
            comp_2_uid = words[3]
            save_path = os.path.join(save_dir, basename + '_' + str(epoch) + '.png')
            save_path_2 = os.path.join(save_dir, basename + '_' + str(epoch) + '_pair.png')
            save_path_comp1 = os.path.join(save_dir, lid + '_' + target_uid + '_' + comp_1_uid +
                                           '_' + str(epoch) + '_comp1.png')
            save_path_comp2 = os.path.join(save_dir, lid + '_' + target_uid + '_' + comp_2_uid +
                                           '_' + str(epoch) + '_comp2.png')
            save_path_trans1 = os.path.join(save_dir, lid + '_' + target_uid + '_' + comp_1_uid + '_' +
                                            str(epoch) + '_trans1.png')
            save_path_trans2 = os.path.join(save_dir, lid + '_' + target_uid + '_' + comp_2_uid + '_' +
                                            str(epoch) + '_trans2.png')
            save_path_output = os.path.join(save_dir, lid + '_' + target_uid + '_' + str(epoch) + '_output.png')
            save_path_target = os.path.join(save_dir, lid + '_' + target_uid + '_' + str(epoch) + '_target.png')

            imageio.imwrite(save_path, result[:, :, 0])
            imageio.imwrite(save_path_2, result_2[:, :, 0])

            imageio.imwrite(save_path_comp1, in_grid1)
            imageio.imwrite(save_path_comp2, in_grid2)
            imageio.imwrite(save_path_trans1, out_grid1)
            imageio.imwrite(save_path_trans2, out_grid2)
            imageio.imwrite(save_path_output, output_grid)
            imageio.imwrite(save_path_target, target_grid)

    def train(self):
        args = self.args
        nets = self.train_net
        optims = self.optim

        # resume training if necessary
        if args.start_iter > 0:
            self.restore_checkpoint(args.start_iter)

        if self.use_cuda:
            self.train_net.cuda()
            self.vgg16.cuda()
        print('Start training...')
        start_time = time.time()
        fetcher = InputFetcher(self.data_loader)
        fetcher_val = InputFetcher(self.val_loader)
        for epoch in range(args.start_iter, args.end_iter):
            # fetch images and labels
            inputs = next(fetcher)
            label = inputs.label
            target = inputs.t_img
            comp_1 = inputs.s_list[0]
            comp_2 = inputs.s_list[1]
            basename = inputs.s_name
            # for batch_idx, data in enumerate(inputs):
            #     label = data[1]
            #     target = data[0][0]
            #     comp_1 = data[0][1][0]
            #     comp_2 = data[0][1][1]
            #     basename = data[0][2]
            if self.use_cuda:
                label, target, comp_1, comp_2 = label.cuda(), target.cuda(), comp_1.cuda(), comp_2.cuda()
            self.optim.zero_grad()
            output, output_1, output_2 = self.train_net(comp_1, comp_2)
            loss_mse = self.criterion(output, target)
            loss_per = self.vgg16(output, target)
            loss = loss_per * 0.1 + loss_mse
            loss.backward()
            self.optim.step()

            if (epoch+1) % args.print_every == 0 or (epoch + 1) == len(self.data_loader):
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, args.end_iters, epoch * self.batch_size, len(self.data_loader.dataset),
                    100. * epoch / len(self.dataloader), loss.data.item()))

                self.save_checkpoint(epoch)
                self.draw_inter(comp_1, comp_2, target, output, output_1, output_2, basename, label, epoch)


if __name__ == '__main__':
    solver = StnSolver(args)
    solver.train()





