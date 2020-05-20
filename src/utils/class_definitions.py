# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/20/2020 12:33 AM
# @Author : Zhuoyi Huang
# @File : class_definitions.py


class CharDef:
    def __init__(self, uid, lid, comps_length, comp_ids, precise_def=False, boxes=None):
        self.uid = uid
        self.lid = lid
        self.compsLen = comps_length
        self.compIds = comp_ids
        self.preciseDef = precise_def
        self.boxes = boxes

    def __str__(self):
        return str((self.uid, self.lid, self.compIds, self.preciseDef))

    def __repr__(self):
        return str(self)

    def set_precise_def(self, boxes):
        self.preciseDef = True
        self.boxes = boxes


class BoundingBox:  # basic characters define
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def __str__(self):
        return str((self.x, self.y, self.dx, self.dy))

    def __repr__(self):
        return str(self)

    def get_size(self):
        return self.dx, self.dy

    def get_box_outline(self):
        return self.x, self.y, self.x + self.dx, self.y + self.dy


class ArrayCollection:  # combined with loss map (not use anymore)
    def __init__(self, uid, style, character=None, rough_definition=None, loss_map=None):
        self.uid = uid
        self.style = style
        self.character = character
        self.roughDefinition = rough_definition
        self.lossMap = loss_map

    def add_character(self, character):
        self.character = character

    def add_rough_def(self, rough_definition):
        self.roughDefinition = rough_definition

    def add_loss_map(self, loss_map):
        self.lossMap = loss_map

    def is_complete(self):
        return self.roughDefinition is not None  # and self.character is not None

    def print_incomplete(self):
        if self.character is None:
            print("Character " + self.uid + " is missing its character array")
        if self.roughDefinition is None:
            print("Character " + self.uid + " is missing its roughDefinition array")
        if self.lossMap is None:
            print("Character " + self.uid + " is missing its lossMap array")

    def as_list(self):
        return [self.uid, self.style, self.character, self.roughDefinition, self.lossMap]
