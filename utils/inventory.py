import cv2 as cv
import pyautogui as pag
from matplotlib import pyplot as plt
import numpy as np


class Inventory:
    '''The inventory is a 4x7 grid of items.
    This class isolates the image of the inventory
    and provides various utilities for processing'''
    def __init__(self, inv_w, inv_h):
        self.inv = None
        self.inv_w = inv_w
        self.inv_h = inv_h

    def update(self, inv):
        self.inv = inv

    def draw_inventory_grid(self):
        '''Illustrates how the image is partitioned into items
        Useful for calibrating the inventory setup on a new monitor'''
        for x in range(0, 5):
            cv.line(self.inv,
                    (self.inv_w*x//4, 0),
                    (self.inv_w*x//4, self.inv_h-1),
                    (255, 0, 0),
                    1)
        for y in range(0, 8):
            cv.line(self.inv,
                    (0, self.inv_h*y//7),
                    (self.inv_w-1, self.inv_h*y//7),
                    (255, 0, 0),
                    1)

    def get_inventory_slot_img(self, i, j):
        '''Extracts the image of a single inventory slot'''
        x_width = self.inv_w//4
        y_width = self.inv_h//7
        top_left = [j*y_width, i*x_width]

        return self.inv[top_left[0]:top_left[0]+y_width,
                        top_left[1]:top_left[1]+x_width]

    def get_inventory_slot_coords(self, i, j, center=True):
        '''Returns the coordinates of an inventory slot
        These are absolute coordinates and ready to be plugged in
        to pyautogui
        CURRENTLY BROKEN'''
        x = (self.screenWidth - self.osrsWidth +
             self.inv_top_left[1] + (self.inv_w*i//4))

        y = (self.screenHeight - self.osrsHeight +
             self.inv_top_left[0] + (self.inv_h*j//7))
        if center:
            x += self.inv_w//(4*2)
            y += self.inv_h//(7*2)
        return x, y

    def check_inventory(self, i, j, obj, threshold=.75):
        '''Checks inventory slot i,j for object.
        Uses template matching with a specified threshold.'''
        inv_slot = self.get_inventory_slot_img(i, j)
        method = cv.TM_CCOEFF_NORMED
        inv_processed = cv.blur(inv_slot, (5, 5))
        obj_processed = cv.blur(obj, (5, 5))
        res = cv.matchTemplate(inv_processed, obj_processed, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if max_val > threshold:
            return True
        return False

    def find_all(self, obj, threshold=.75):
        '''returns a list of inventory slots for items matching the template'''
        results = []
        for j in range(0, 7):
            for i in range(0, 4):
                if self.check_inventory(i, j, obj, threshold=threshold):
                    results.append([i, j])
        return results
