import yaml
import cv2 as cv
import numpy as np
import torch
import os

from src.Demos import UtilDemos

from src.Segmentation.CollectTrainingData import TrainingDataGenerator
from src.Segmentation import CNN
from src.Segmentation import Segment
from src.FutureFramePrediction import CNN3D


def CreateTrainingData():
    '''
    Capture samples around the mouse and generates
    labels using the tooltip.
    '''
    tdh = TrainingDataGenerator(base_dir='samples_{}.pkl')
    tdh.CollectAndPickleSamples(config_dir='screen_cfg.yaml',
                                num_samples=200,
                                batch_size=100,
                                box_l=32,
                                delay=.05,
                                do_print=True)
    tdh.LabelPickledSamples(config_dir='screen_cfg.yaml')
    tdh.PicklesToImageDataset(img_dir='data/images/{}')


def TrainCNNDemo():
    CNN.main()


def Train3DCNNDemo():
    CNN3D.main()


def ParseScreenWithCNN(num_classes):
    Segment.segment_screen_live(num_classes=num_classes, diffuse_time=False)


def CompareSpatialTemporalFiltering():
    # RecordAndSaveVideo('data/videos/cows_near/{}.png')
    # Segment.predict_directory('data/videos/cows_near', 'data/videos/cows_near_pred.pkl', num_classes=2)
    # Segment.predict_directory('data/videos/cows_near', 'data/videos/cows_near_spatial1_pred.pkl', diffuse_space=True, num_classes=2)
    # Segment.predict_directory('data/videos/cows_near', 'data/videos/cows_near_temporal3_pred.pkl', diffuse_time=True, num_classes=2)
    # SegmentWithWithCNN(num_classes=2)
    frames1 = Segment.render_directory('data/videos/cows_near', 'data/videos/cows_near_pred.pkl')
    frames2 = Segment.render_directory('data/videos/cows_near', 'data/videos/cows_near_spatial1_pred.pkl')
    frames3 = Segment.render_directory('data/videos/cows_near', 'data/videos/cows_near_temporal3_pred.pkl')
    orig_vs_space = [np.hstack((frame1, frame2)) for (frame1, frame2) in zip(frames1, frames2)]
    orig_vs_time = [np.hstack((frame1, frame3)) for (frame1, frame3) in zip(frames1, frames3)]        
    h, w = orig_vs_space[0].shape[:2]
    out1 = cv.VideoWriter('data/videos/orig_vs_space_near.mp4', cv.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
    out2 = cv.VideoWriter('data/videos/orig_vs_time_near.mp4', cv.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
    for idx in range(len(frames1)):
        cv.imshow('Original Pred', frames1[idx])
        cv.imshow('Spatially Diffused Pred', frames2[idx])
        cv.imshow('Temporally Averaged Pred', frames3[idx])
        cv.waitKey(1)
        out1.write(orig_vs_space[idx])
        out2.write(orig_vs_time[idx])
    out1.release()
    out2.release()


if __name__ == '__main__':
    # UtilDemos.ShowROIs()
    # UtilDemos.CaptureAndEditVideo()
    # UtilDemos.ExtractTooltips()
    # UtilDemos.RecordAndSaveVideo('data/videos/cows_test/{}.png')
    
    # CreateTrainingData()
    # TrainMobileDemo()
    # ParseScreenWithMobileNet()
    TrainCNNDemo()
    ParseScreenWithCNN(num_classes=2)
    # CompareSpatialTemporalFiltering()

    # ParseScreenCNN.predict_directory('data/videos/cows_test', 'data/videos/cows_test_pred.pkl', num_classes=2)
    # Train3DCNNDemo()
    # Train3DCNN.load_and_test()

    '''
    frames_raw = []
    frame_files = ParseScreenCNN.natural_sort(os.listdir('data/videos/cows_near'))
    for idx, frame_file in enumerate(frame_files):
        frames_raw.append(cv.imread('data/videos/cows_near' + '/' + frame_file))
    frames_labeled = ParseScreenCNN.render_directory('data/videos/cows_near', 'data/videos/cows_near_pred.pkl')
    for idx in range(0, len(frames_raw)):
        cv.imshow('Original', frames_raw[idx])
        cv.imshow('Labeled', frames_labeled[idx])
        cv.waitKey(0)
    '''
