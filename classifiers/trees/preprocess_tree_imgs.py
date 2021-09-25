import cv2 as cv

def preprocess_all_images(cur_dir, filename, num, startnum=1):
    scale_percent = 80 # percent of original size
    for x in range(startnum, num+1):
        bw_img = cv.imread(cur_dir + filename.format(x), 0)
        width = int(bw_img.shape[1] * scale_percent / 100)
        height = int(bw_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        bw_img = cv.resize(bw_img, dim, interpolation=cv.INTER_AREA)
        cv.imwrite(cur_dir + 'processed/' + filename.format(x), bw_img)

#preprocess_all_images('/home/captdishwasher/horenbergerb/OpenCV/osrs/images/trees/positives/', 'trees{}.png', 11)

#preprocess_all_images('/home/captdishwasher/horenbergerb/OpenCV/osrs/images/trees/negatives/', 'not_trees{}.png', 11)

#preprocess_all_images('/home/captdishwasher/horenbergerb/OpenCV/osrs/images/trees/test/', 'test{}.png', 2)

preprocess_all_images('/home/captdishwasher/horenbergerb/OpenCV/osrs/images/trees/new_positives/', 'trees{}.png', 24, startnum=14)
