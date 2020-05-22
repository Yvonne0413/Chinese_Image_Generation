# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/19/2020 12:36 PM
# @Author : Zhuoyi Huang
# @File : generate_character_pairs.py

# this file is used to generate character pairs regarding the components of a specific character

import os
import sys
import argparse
import utils.sqlite_database as sqlite_database
from PIL import Image
from utils.class_definitions import BoundingBox, CharDef, ArrayCollection
from utils.layouts import layouts, applyLayout, applyPreciseLayout

os.environ['KMP_DUPLICATE_LIB_OK']='True'  # for Mac OS users

# possible arguments
# fonts style name
# styles = ["Heiti", "Kaiti_1", "Kaiti_2", "Regular", "Songti_1", "Songti_2"]
# SPECIAL_UIDS = ["673a", "5668", "5b66", "4e60",
#                 "4e2d", "6587", "5b57", "4f53",
#                 "63a2", "7d22", "6280", "672f",
#                 "8bbe", "8ba1", "672a", "6765"]

parser = argparse.ArgumentParser(description='Generate character pairs')
parser.add_argument('--input_dir', dest='input_dir', default="../data/interFiles/FontImages/",
                    help='base directory for fonts image, include images with 1000*1000 size')
parser.add_argument('--style_name', dest='style_name', default="Regular",
                    help='style name, default Regular')
parser.add_argument('--rough_dir_name', dest='rough_dir_name', default="comp_rough_256",
                    help='rough_dir_name, default comp_rough_256')
parser.add_argument('--unit_size', dest='unit_size', default=256,
                    help='unit_size of each photo in a pair, default 256')
parser.add_argument('--quality_value', dest='quality_value', default=100,
                    help='quality value of images to save')
parser.add_argument('--special_uids', dest='special_uids', default=[],
                    help='special_uids')
parser.add_argument('--img_size', dest='img_size', default=1000,
                    help='img size')
args = parser.parse_args()


class GeneratePairs:
    def __init__(self, input_dir, style_name, rough_dir_name, unit_size, quality_value, special_uids, img_size):
        self.input_dir = input_dir
        self.style_name = style_name
        self.rough_dir_name = rough_dir_name
        self.unit_size = unit_size
        self.quality_value = quality_value
        self.special_uids = special_uids
        self.img_size = img_size
        self.char_def_dic = {}
        self.rough_image_dic = {}
        self.array_dic = {}

        if input_dir:
            self.all_path_1000 = os.path.join(input_dir, style_name, "all_1000/")
            self.origin_path_1000 = os.path.join(input_dir, style_name, "origin_1000/")
            self.rough_path_1000 = os.path.join(input_dir, style_name, "rough_1000/")
            self.origin_path_256 = os.path.join(input_dir, style_name, "origin_256/")
            self.rough_path_256 = os.path.join(input_dir, style_name, "rough_256/")
            self.comp_path_256 = os.path.join(input_dir, style_name, "comp_256/")

        if not os.path.exists(self.origin_path_1000):
            os.makedirs(self.origin_path_1000)
            print("create origin_path_1000 directory")

        if not os.path.exists(self.rough_path_1000):
            os.makedirs(self.rough_path_1000)
            print("create rough_path_1000 directory")

        if not os.path.exists(self.origin_path_256):
            os.makedirs(self.origin_path_256)
            print("create origin_path_1000 directory")

        if not os.path.exists(self.rough_path_256):
            os.makedirs(self.rough_path_256)
            print("create rough_path_256 directory")

        if not os.path.exists(self.comp_path_256):
            os.makedirs(self.comp_path_256)
            print("create comp_path_256 directory")

    def set_char_def_dic(self, char_def_dic):
        self.char_def_dic = char_def_dic

    def set_rough_image_dic(self, rough_image_dic):
        self.rough_image_dic = rough_image_dic

    def set_array_dic(self, array_dic):
        self.array_dic = array_dic

    def get_base_images(self):
        print("Adding base images to dictionary")
        rough_image_dic = {}
        array_dic = {}
        for file in os.listdir(self.origin_path_256):
            if file.endswith(".png"):
                uid = file[:-4]
                char_def = self.char_def_dic.get(uid)
                if char_def is not None:
                    try:
                        image = Image.open(self.origin_path_256 + file)
                        image = image.convert("1")
                        image = image.resize((self.unit_size, self.unit_size))
                        if char_def.lid == 0:
                            rough_image_dic.update({uid: image})
                        else:
                            ac = ArrayCollection(uid, self.style_name, image)
                            array_dic.update({uid: ac})
                        # print(rough_image_dic)
                        # print(array_dic)
                    except Exception as error:
                        ac = ArrayCollection(uid, self.style_name)
                        array_dic.update({uid: ac})
                        # print("Error encountered for unicode: ", uid)
                        # print("Error message: ", error)
        return rough_image_dic, array_dic

    def create_char_image(self, uid):
        """
        for each character(uid), get its images. given it is not contained in the dict
        :param uid:
        :return:
        """
        try:
            c_def = self.char_def_dic.get(uid)
            lid = c_def.lid
            comp_ids = c_def.compIds
            precise_def = c_def.preciseDef
            boxes1000 = c_def.boxes
        except Exception as error:
            print("Error encountered for unicode: ", c_def.uid)
            print("create CharImage Error message: ", error)
        if lid == 0:
            raise Exception("Base image for character id " + uid + " was not provided")
        else:
            comp_imgs = []  # components list
            for i in comp_ids:
                if i not in self.rough_image_dic.keys():
                    self.create_char_image(i)
                img = self.rough_image_dic.get(i)
                comp_imgs.append(img)
            if precise_def:
                boxes = []
                for box1000 in boxes1000:
                    x = int(box1000.x * self.unit_size / 256)
                    y = int(box1000.y * self.unit_size / 256)
                    dx = int(box1000.dx * self.unit_size / 256)
                    dy = int(box1000.dy * self.unit_size / 256)
                    box = BoundingBox(x, y, dx, dy)
                    boxes.append(box)
                im = applyPreciseLayout(boxes, comp_imgs)
            else:
                im = applyLayout(lid, comp_imgs)
            self.rough_image_dic.update({uid: im})
            self.save_component(uid, lid, comp_imgs, comp_ids)

    def save_component(self, uid, lid, comp_imgs, comp_ids):
        print("save the components images for one character")
        save_dir = os.path.join(self.comp_path_256, "lid" + str(lid) + "/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        comps = layouts.get(lid, layouts.get(0))
        if len(comp_imgs) != len(comps):
            print("incorrect lengths")
            # Throw error

        else:
            target_width = (len(comps) + 1) * self.unit_size
            target_pair = Image.new('1', (target_width, self.unit_size))
            left, right = 0, self.unit_size
            origin_image = Image.open(self.origin_path_256 + uid + '.png')
            target_pair.paste(origin_image, (left, 0, right, self.unit_size))
            left += self.unit_size
            right += self.unit_size
            save_name = "lid" + str(lid) + "_" + "src_" + str(uid).zfill(5)
            for i in range(len(comp_imgs)):
                box = (left, 0, right, self.unit_size)
                target_pair.paste(comp_imgs[i], tuple(box))
                left += self.unit_size
                right += self.unit_size
                save_name = save_name + "_" + str(comp_ids[i]).zfill(5)
            # print("target_pair:", target_pair)]
            if uid in self.special_uids:
                print("save target pair")
                target_pair.save(save_dir + save_name + '_s.png', quality=self.quality_value)
            else:
                print("save target pair")
                target_pair.save(save_dir + save_name + '.png', quality=self.quality_value)

        return target_pair

    def add_rough_def_arrays(self):
        print("Creating new character images")
        i = 0
        tot = len(self.char_def_dic)
        for uid, char_def in self.char_def_dic.items():
            i += 1
            perc = i * 100 // tot
            sys.stdout.write("\r%i %%" % perc)
            sys.stdout.flush()
            if char_def.lid != 0:
                try:
                    if uid not in self.rough_image_dic.keys():  # not exist in the char dic
                        self.create_char_image(uid)
                    rough_image = self.rough_image_dic.get(uid)
                    ac = self.array_dic.get(uid)
                    ac.add_rough_def(rough_image)
                except Exception as error:
                    # pass
                    print("Error encountered for unicode: ", uid)
                    print("addRoughDefArrays Error message: ", error)

    def generate_pairs(self):
        print("Importing arrays for style:", self.style_name)
        self.char_def_dic = sqlite_database.get_char_def_dic()
        # import from utils create a dictionary about all characters
        self.rough_image_dic, self.array_dic = self.get_base_images()

        # self.set_rough_image_dic(rough_image_dic)
        # self.set_array_dic(array_dic)

        self.add_rough_def_arrays()
        print("Done")


if __name__ == '__main__':
    generator = GeneratePairs(args.input_dir, style_name=args.style_name, rough_dir_name=args.rough_dir_name,
                              unit_size=args.unit_size, quality_value=args.quality_value,
                              special_uids=args.special_uids, img_size=args.img_size)
    generator.generate_pairs()