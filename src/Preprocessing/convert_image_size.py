# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/20/2020 5:37 AM
# @Author : Zhuoyi Huang
# @File : convert_img_size.py

from PIL import Image
import os
import glob
import argparse


parser = argparse.ArgumentParser(description='Convert_image size')
parser.add_argument('--in_path', dest='in_path', default="../../data/interFiles/FontImages/",
                    help='fonts file directory, include files end with otf or ttf.')
parser.add_argument('--out_path', dest='out_path', default="../../data/interFiles/FontImages/",
                    help='font out directory')
parser.add_argument('--style_name', dest='style_name', default="Shouxie",
                    help='fonts style name, default regular')
parser.add_argument('--origin_size', dest='origin_size', default=1000,
                    help='origin image size')
parser.add_argument('--target_size', dest='target_size', default=256,
                    help='target image size')
args = parser.parse_args()


class ConvertImage:
    def __init__(self, in_path, out_path, style_name, origin_size, target_size):
        self.in_path = in_path
        self.out_path = out_path
        self.style_name = style_name
        self.origin_size = origin_size
        self.target_size = target_size

        self.rough_convert_in_path = os.path.join(in_path, style_name, "rough_1000/") + "*.png"
        self.rough_convert_out_path = os.path.join(out_path, style_name, "rough_256/")

        self.origin_convert_in_path = os.path.join(in_path, style_name, "all_1000/") + "*.png"
        self.origin_convert_out_path = os.path.join(out_path, style_name, "origin_256/")

        if not os.path.exists(self.origin_convert_out_path):
            os.makedirs(self.origin_convert_out_path)

        if not os.path.exists(self.rough_convert_out_path):
            os.makedirs(self.rough_convert_out_path)

    def convert_size(self, img_file, out_path):
        img = Image.open(img_file)
        try:
            new_img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
            new_img.save(os.path.join(out_path, os.path.basename(img_file)))
        except Exception as error:
            print(error)

    def convert_all_img(self, in_path, out_path):
        print("convert all img")
        print(in_path)
        for img_file in glob.glob(in_path):
            self.convert_size(img_file, out_path)

    def convert_dir_img(self):
        self.convert_all_img(self.rough_convert_in_path, self.rough_convert_out_path)
        self.convert_all_img(self.origin_convert_in_path, self.origin_convert_out_path)


if __name__ == '__main__':
    covert = ConvertImage(in_path=args.in_path, out_path=args.out_path, style_name=args.style_name,
                          origin_size=args.origin_size, target_size=args.target_size)
    covert.convert_dir_img()

