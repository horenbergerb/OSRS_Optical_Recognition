import yaml
import cv2 as cv
import numpy as np
import torch
import os

from src.utils.ScreenManager import ScreenManager
from src.utils.VideoTools import VideoCapturer, VideoEditor
from src.utils.ExtractionTools import extract_colors
from src.utils.PickleUtils import LoadPickle
from src.Segmentation.CollectTrainingData import TrainingDataGenerator
from src.Segmentation import MobileNet
from src.Segmentation import SegmentWithMobileNet
from src.Segmentation import CNN
from src.Segmentation import SegmentWithCNN
from src.FutureFramePrediction.FramePredictionDataset import FramePredictionDataset
from src.FutureFramePrediction import CNN3D


def ShowROIs():
    config = yaml.safe_load(open('screen_cfg.yaml'))
    sm = ScreenManager(screen_dims=config['screen'], regions=config['regions'])
    while True:
        sm.update()
        cv.imshow('Annotated Screen', sm.annotated_screen())
        cv.waitKey(10)


def CaptureAndEditVideo():
    config = yaml.safe_load(open('screen_cfg.yaml'))
    sm = ScreenManager(screen_dims=config['screen'], regions=config['regions'])
    ah = VideoCapturer(sm)
    frames = ah.capture()
    frames = VideoEditor(frames).playback()


def RecordAndSaveVideo(save_dir):
    # currently only saves playscreen
    config = yaml.safe_load(open('screen_cfg.yaml'))
    s_x = config['regions']['playscreen']['x']
    s_y = config['regions']['playscreen']['y']
    s_w = config['regions']['playscreen']['w']
    s_h = config['regions']['playscreen']['h']

    sm = ScreenManager(screen_dims=config['screen'], regions=config['regions'])
    ah = VideoCapturer(sm)
    frames = ah.capture(delay=18)
    for idx, frame in enumerate(frames):
        cv.imwrite(
            save_dir.format(idx),
            frame.img[s_y:s_y+s_h, s_x:s_x+s_w, :])


def ExtractTooltips():
    config = yaml.safe_load(open('screen_cfg.yaml'))
    sm = ScreenManager(screen_dims=config['screen'], regions=config['regions'])
    while True:
        sm.update()
        for color_name, colors in config['text_colors'].items():
            mask = extract_colors(sm.tooltip, colors, 50)
            cv.imshow('Tooltip', sm.tooltip)
            cv.imshow('{} Mask'.format(color_name), mask.astype(float))
        cv.waitKey(10)


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
    # samples = LoadPickle('samples_1.pkl')
    # for cur in samples:
    #     cv.imshow('Image', cur.img)
    #     cv.imshow('Tooltip', cur.tooltip)
    #     print('Label: {}'.format(cur.label))
    #     cv.waitKey(0)
    tdh.LabelPickledSamples(config_dir='screen_cfg.yaml')
    tdh.PicklesToImageDataset(img_dir='data/images/{}')


def TrainMobileDemo():
    MobileNet.main()


def ParseScreenWithMobileNet():
    SegmentWithMobileNet.parse_screen_live()


def TrainCNNDemo():
    CNN.main()


def Train3DCNNDemo():
    CNN3D.main()


def ParseScreenWithCNN(num_classes):
    SegmentWithCNN.parse_screen_live(num_classes=num_classes, diffuse_time=False)


def TestFramePredDataLoader():
    device = torch.device("cuda")
    dataset = FramePredictionDataset('data/videos/cows_near_pred.pkl', device)
    print(dataset[0]['input'].shape)
    print(dataset[0]['label'].shape)


def CompareSpatialTemporalFiltering():
    # RecordAndSaveVideo('data/videos/cows_near/{}.png')
    # SegmentWithCNN.predict_for_directory('data/videos/cows_near', 'data/videos/cows_near_pred.pkl', num_classes=2)
    # SegmentWithCNN.predict_for_directory('data/videos/cows_near', 'data/videos/cows_near_spatial1_pred.pkl', diffuse_space=True, num_classes=2)
    # SegmentWithCNN.predict_for_directory('data/videos/cows_near', 'data/videos/cows_near_temporal3_pred.pkl', diffuse_time=True, num_classes=2)
    # SegmentWithWithCNN(num_classes=2)
    frames1 = SegmentWithCNN.render_directory('data/videos/cows_near', 'data/videos/cows_near_pred.pkl')
    frames2 = SegmentWithCNN.render_directory('data/videos/cows_near', 'data/videos/cows_near_spatial1_pred.pkl')
    frames3 = SegmentWithCNN.render_directory('data/videos/cows_near', 'data/videos/cows_near_temporal3_pred.pkl')
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


def RenderLabels():
    color_filters = [np.full((32, 32, 3), (0, 0, 255), np.uint8),
         np.full((32, 32, 3), (245, 66, 114), np.uint8),
         np.full((32, 32, 3), (0, 255, 0), np.uint8),
         np.full((32, 32, 3), (255, 0, 0), np.uint8),
         np.full((32, 32, 3), (255, 255, 0), np.uint8),
         np.full((32, 32, 3), (0, 255, 255), np.uint8),
         np.full((32, 32, 3), (255, 0, 255), np.uint8)]
    predictions = LoadPickle('data/videos/cows_near_pred.pkl')
    for prediction in predictions:
        labels = torch.argmax(prediction, dim=-1)
        raw_frame = np.zeros((320, 512, 3),dtype=np.uint8)
        for y in range(0, raw_frame.shape[1], 32):
            for x in range(0, raw_frame.shape[0], 32):
                raw_frame[x:x+32, y:y+32] = color_filters[labels[x//32, y//32]]
        cv.imshow('Predictions', raw_frame)
        cv.waitKey(0)


if __name__ == '__main__':
    # ShowROIs()
    # CaptureAndEditVideo()
    # ExtractTooltips()
    # CreateTrainingData()
    # TrainMobileDemo()
    # ParseScreenWithMobileNet()
    # TrainCNNDemo()
    # ParseScreenWithCNN(num_classes=2)
    # CompareSpatialTemporalFiltering()
    # RecordAndSaveVideo('data/videos/cows_test/{}.png')
    # ParseScreenCNN.predict_for_directory('data/videos/cows_test', 'data/videos/cows_test_pred.pkl', num_classes=2)
    # TestFramePredDataLoader()
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
