import cv2 as cv
from matplotlib import pyplot as plt


def find_on_map(template):
    world_map = cv.imread('images/osrs_world_map.png', 0)

    # scaling so the proportion approximately matches our world map resolution
    scale_percent = 75
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    dim = (width, height)
    template = cv.resize(template, dim, interpolation=cv.INTER_AREA)

    method = cv.TM_CCOEFF_NORMED
    world_map_processed = cv.blur(world_map, (5, 5))
    template_processed = cv.blur(template, (5, 5))
    res = cv.matchTemplate(template_processed, world_map_processed, method)

    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res, None)

    print('Maximum match found at {}'.format(maxLoc))
    cv.rectangle(world_map, maxLoc, (maxLoc[0] + template.shape[1], maxLoc[1] + template.shape[0]), (0, 0, 0), 2, 8, 0)

    plt.subplot(131), plt.imshow(world_map, cmap='gray')
    plt.subplot(132), plt.imshow(res, cmap='gray')
    plt.subplot(133), plt.imshow(template, cmap='gray')
    plt.show()


if __name__ == '__main__':
    template = cv.imread('images/lumbridge_mini_map1.png', 0)
    find_on_map(template, minimap=True)

    template = cv.imread('images/lumbridge_mini_map2.png', 0)
    find_on_map(template, minimap=True)
