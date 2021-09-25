import cv2 as cv
import mss
import yaml
import numpy as np
from utils.inventory import Inventory

config = yaml.safe_load(open('config.yaml'))
screen_cfg = config['screen']
inventory_cfg = config['inventory']


def parse_inventory(sct, inventory, obj, item_name='item', prev_result=None):
    img = np.array(
        sct.grab({'top': screen_cfg['screen_top']+inventory_cfg['inventory_top'],
                  'left': screen_cfg['screen_left']+inventory_cfg['inventory_left'],
                  'height': inventory_cfg['inventory_height'],
                  'width': inventory_cfg['inventory_width']}))
    inventory.update(img)
    result = inventory.find_all(obj, threshold=.7)
    if result != prev_result:
        print('Search for {} returned positive for the following slots: {}'.format(
            item_name, result))

    inventory.draw_inventory_grid()
    cv.imshow('Parsed Inventory', inventory.inv)
    cv.waitKey(1)
    return result


if __name__ == '__main__':
    inventory = Inventory(inventory_cfg['inventory_width'], inventory_cfg['inventory_height'])
    obj = cv.imread('images/logs.png')
    obj = cv.cvtColor(obj, cv.COLOR_BGR2RGBA)
    with mss.mss() as sct:
        result = None
        while True:
            result = parse_inventory(sct, inventory, obj, item_name='Oak logs', prev_result=result)
            break
