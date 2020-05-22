# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/19/2020 12:36 PM
# @Author : Zhuoyi Huang
# @File : create_character_images.py

# this file is used to generate 1000*1000 character images(unicode from 4e00 to 99999) for a specific font

import os
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageChops
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # for Mac OS users

# possible arguments

# fonts file
fonts = ["HYQiHei-25JF.otf", "HanYiXiXingKaiJian-1.ttf", "Kaiti_2.TTF", "SourceHanSansCN-Regular.otf",
         "HYXinRenWenSongW-1.otf", "SourceHanSerifCN-Heavy-4.otf", "XingshuShouxie.ttf"]

# fonts style name
styles = ["Heiti", "Kaiti", "Kaiti", "Regular", "Songti", "Songti", "Shouxie"]

# additional style name
adds = ["_2", "_1", "_2", "", "_1", "_2", ""]

# fonts offset
offsets = [-70, 0, -20, 0, -60, -260]

# image sizes
sizes = 1000


parser = argparse.ArgumentParser(description='Generate character images')
parser.add_argument('--fonts_dir', dest='fonts_dir', default="../../data/inputFiles/FontsFiles/",
                    help='fonts file directory, include files end with otf or ttf.')
parser.add_argument('--font_filename', dest='font_filename', default="XingshuShouxie.ttf",  # modified
                    help='font file name, default SourceHanSansCN-Regular.otf')
parser.add_argument('--style_name', dest='style_name', default="Shouxie",   # modified
                    help='fonts style name, default regular')
parser.add_argument('--add_name', dest='add_name', default="",  # modified
                    help='fonts add name, default ')
parser.add_argument('--y_offset', dest='y_offset', default=0,    # modified,
                    help='fonts offsets, default 0')
parser.add_argument('--size', dest='size', default=1000,
                    help='generate image size default 1000')
parser.add_argument('--output_dir', dest='output_dir', default="../../data/interFiles/FontImages/",
                    help='save dir default "../../data/interFiles/FontImages/"')
args = parser.parse_args()


class GenerateImages:
    def __init__(self, fonts_dir, font_filename, style_name, add_name, y_offset, size, output_dir):
        self.fonts_dir = fonts_dir
        self.font_filename = font_filename
        self.style_name = style_name
        self.add_name = add_name
        self.y_offset = y_offset
        self.size = size
        self.output_dir = output_dir

        if fonts_dir:
            self.input_path = os.path.join(fonts_dir, style_name, font_filename)
            self.output_dir = os.path.join(output_dir, style_name + add_name, "all_1000")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("create output directory")

    @staticmethod
    def take_unicode(char):
        """
        get the unicode string of a character
        :param char: char input character
        :return: string unicode
        """
        return char.encode('unicode_escape').decode()

    @staticmethod
    def is_empty(im):
        """
        if empty
        :param im:
        :return:
        """
        return not ImageChops.invert(im).getbbox()

    def get_font_image(self, char):
        """
        get the font image for a specific character
        :param char: input char
        :return: Image image of the character
        """
        im = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(self.input_path, self.size)
        dr.text((0, self.y_offset), char, font=font, fill="#000000")
        im = im.convert('1')
        return im

    def generate_image(self):
        """
        generate images for the file
        :return:
        """
        characters = []  # character list
        for i in range(0x4E00, 0x9FA5 + 1):
            characters.append(chr(i))
        write_directory = self.output_dir

        print("Creating images for" + self.style_name + self.add_name)
        print("Images saved to" + write_directory)
        for char in characters:
            unicode = self.take_unicode(char)[2:]
            try:
                im = self.get_font_image(char)
                if self.is_empty(im):
                    pass
                    print("Character missing for font: " + self.style_name + "/" + self.font_filename)
                    print("Unicode: ", unicode)
                else:
                    im.save(write_directory+'/{}.png'.format(unicode))
            except Exception as error:
                pass
                print("Exception: ", error)
                print("For unicode: ", unicode)


if __name__ == '__main__':
    generator = GenerateImages(args.fonts_dir, font_filename=args.font_filename, style_name=args.style_name,
                               add_name=args.add_name, y_offset=args.y_offset, size=args.size,
                               output_dir=args.output_dir)
    generator.generate_image()




