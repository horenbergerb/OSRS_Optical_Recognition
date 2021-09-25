import cv2 as cv
import os


def preprocess_all_images(cur_dir, scale_percent=80):
    files = os.listdir(cur_dir)
    for f in files:
        if f == 'processed':
            continue
        bw_img = cv.imread(cur_dir + f, 0)
        width = int(bw_img.shape[1] * scale_percent / 100)
        height = int(bw_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        bw_img = cv.resize(bw_img, dim, interpolation=cv.INTER_AREA)
        cv.imwrite(cur_dir + 'processed/' + f, bw_img)


if __name__ == '__main__':
    preprocess_all_images('positives/')
    preprocess_all_images('negatives/')
    preprocess_all_images('test/')
    preprocess_all_images('new_positives/')
