import cv2 as cv
import os

cascade = cv.CascadeClassifier()
if not cascade.load(cv.samples.findFile('classifier/cascade.xml')):
    print('Error loading cascade')
    exit(0)

test_dir = 'test/processed'
for f in os.listdir(test_dir):
    img_test = cv.imread(test_dir+'/'+f, 0)
    cv.imshow('Original Image', img_test)
    detections = cascade.detectMultiScale(img_test, scaleFactor=1.05, minNeighbors=4)
    for (x, y, w, h) in detections:
        center = (x + w//2, y + h//2)
        img_test = cv.rectangle(img_test, (x, y), (x+w, y+h), (255, 0, 255), 2)
    cv.imshow('Detections', img_test)
    cv.waitKey(0)
