# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/20/2020 7:16 AM
# @Author : Zhuoyi Huang
# @File : layouts.py

from PIL import Image
import numpy as np
from .class_definitions import BoundingBox
from .constants import layouts

IMAGES_WIDTH = 1000
IMAGES_HEIGHT = 1000
IMAGES_SIZE = (IMAGES_WIDTH, IMAGES_HEIGHT)


# input an image finds the smallest bounding box in the initial image to a size (rid the white space)
def findBoundingBox(image):
    img = np.invert(np.asarray(image))
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return BoundingBox(xmin, ymin, xmax-xmin, ymax-ymin)


def findCutout(targetB, charB): # helper find the largest box contains the character
    if targetB.dx == IMAGES_WIDTH:
        x = 0
        dx = IMAGES_WIDTH
    else:
        x = charB.x
        dx = charB.dx
        while dx < targetB.dx - 1 and x > 0 and x < IMAGES_WIDTH-1:
            x = x-1
            dx = dx+2
        if dx == targetB.dx - 1 and x > 0:
            x = x-1
            dx = dx+1
        elif dx == targetB.dx - 1:
            dx = dx+1
        if x == 0 and dx < targetB.dx:
            dx = targetB.dx
        elif x == IMAGES_WIDTH-1 and dx < targetB.dx:
            dx = targetB.dx
            x = IMAGES_WIDTH-1 - dx
    if targetB.dy == IMAGES_HEIGHT:
        y = 0
        dy = IMAGES_HEIGHT
    else:
        y = charB.y
        dy = charB.dy
        while dy < targetB.dy - 1 and y > 0 and y < IMAGES_HEIGHT-1:
            y = y-1
            dy = dy+2
        if dy == targetB.dy - 1 and y > 0:
            y = y-1
            dy = dy+1
        elif dy == targetB.dy -1:
            dy = dy+1
        if y == 0 and dy < targetB.dy:
            dy = targetB.dy
        elif y == IMAGES_HEIGHT-1 and dy < targetB.dy:
            dy = targetB.dy
            y = IMAGES_HEIGHT-1 - dy
    return x, y, x+dx, y+dy


def needsResizing(targetB, charB): # helper for the layout
    return targetB.dx < charB.dx or targetB.dy < charB.dy


def applyLayout(layout, images): # create rough dataset
    comps = layouts.get(layout, layouts.get(0))
    if len(images) != len(comps):
        print("incorrect lengths")
        # Throw error
    else:
        outImage = Image.new("1", IMAGES_SIZE, 1)
        for i in range(len(images)):
            comp = comps[i]
            image = images[i]

            targetBox = BoundingBox(int(comp[0]*IMAGES_WIDTH), int(comp[1]*IMAGES_HEIGHT), int(comp[2]*IMAGES_WIDTH), int(comp[3]*IMAGES_HEIGHT))
            charBox = findBoundingBox(image)
            if needsResizing(targetBox, charBox):
                tempImage = image.resize(targetBox.get_size())
            else:
                cutout = findCutout(targetBox, charBox)
                tempImage = image.crop(cutout)
                # tempImage = image.resize(targetBox.getSize())
            outImage.paste(tempImage, targetBox.get_box_outline(), outImage.crop(targetBox.get_box_outline()))
    return outImage


def applyPreciseLayout(boxes, images):
    if len(images) != len(boxes):
        print("incorrect lengths")
        # Throw error
    else:
        outImage = Image.new("1", IMAGES_SIZE, 1)
        for i in range(len(images)):
            image = images[i]
            targetBox = boxes[i]
            charBox = findBoundingBox(image)
            char = image.crop(charBox.get_box_outline()).resize(targetBox.get_size())
            outImage.paste(char, targetBox.get_box_outline(), outImage.crop(targetBox.get_box_outline()))
    return outImage