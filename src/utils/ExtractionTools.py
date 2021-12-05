import numpy as np


def extract_colors(img, colors, tolerance):
    '''Returns a mask for the pixels which are  within the allowed tolerance
    for at least one of the provided colors'''
    mask = np.zeros(img.shape[:-1]).astype(bool)
    for idx, color in enumerate(colors):
        dists = np.sum(np.abs(img.astype(int) - color), axis=-1)
        mask = np.logical_or(mask, np.less(dists, tolerance))
    return mask
