import cv2 as cv
import mss
import yaml
import numpy as np

config = yaml.safe_load(open('config.yaml'))
screen_cfg = config['screen']

cascade = cv.CascadeClassifier()
if not cascade.load(cv.samples.findFile(config['classifier']['classifier_dir'])):
    print('Error loading cascade')
    exit(0)


def get_detections(sct):
    img = np.array(
        sct.grab({'top': screen_cfg['screen_top'],
                  'left': screen_cfg['screen_left'],
                  'height': screen_cfg['play_screen_height'],
                  'width': screen_cfg['play_screen_width']}))
    detections = cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=4)
    img = img
    for detection in detections:
        cv.rectangle(img,
                     (detection[0], detection[1]),
                     (detection[0] + detection[2], detection[1] + detection[3]),
                     (255, 0, 0),
                     2)
    cv.imshow('Detections', img)
    cv.waitKey(1)


if __name__ == '__main__':
    with mss.mss() as sct:
        while True:
            get_detections(sct)
