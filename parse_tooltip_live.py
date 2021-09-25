import cv2 as cv
import mss
import yaml
import numpy as np
from pytesseract import image_to_string

config = yaml.safe_load(open('config.yaml'))
screen_cfg = config['screen']
tooltip_cfg = config['tooltip']

kernel = np.ones((5, 5), np.uint8)


def make_bw_and_scale(img, scale_percent=600):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def parse_tooltip(sct):
    img = np.array(
        sct.grab({'top': screen_cfg['screen_top']+tooltip_cfg['tooltip_top'],
                  'left': screen_cfg['screen_left']+tooltip_cfg['tooltip_left'],
                  'height': tooltip_cfg['tooltip_height'],
                  'width': tooltip_cfg['tooltip_width']}))
    img = make_bw_and_scale(img)
    img = cv.medianBlur(img, 3)
    # img = cv.inRange(img, (50, 30, 0), (255, 255, 100))
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    img = 255-img
    img = cv.dilate(img, kernel, iterations=1)
    img = cv.erode(img, kernel, iterations=1)
    # img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img = cv.GaussianBlur(img, (5, 5), 0)
    cv.imshow('Processed Tooltip', img)
    cv.waitKey(1)
    print(image_to_string(img, lang='eng', config='--psm 7'))


if __name__ == '__main__':
    with mss.mss() as sct:
        while True:
            parse_tooltip(sct)
