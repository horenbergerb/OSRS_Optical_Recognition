import yaml
import cv2 as cv

from src.utils.ScreenManager import ScreenManager
from src.utils.VideoTools import VideoCapturer, VideoEditor
from src.utils.ExtractionTools import extract_colors
from src.MobileNet.MobileTrainingData import TrainingDataGenerator, LoadPickle
from src.MobileNet import TrainMobileNet


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
                                num_samples=400,
                                batch_size=200,
                                box_l=32,
                                delay=.05,
                                do_print=True)
    samples = LoadPickle('samples_1.pkl')
    labels = LoadPickle('samples_labels.pkl')
    print(labels)
    for cur in samples:
        cv.imshow('Image', cur.img)
        cv.imshow('Tooltip', cur.tooltip)
        print('Label: {}'.format(cur.label))
        cv.waitKey(0)
    tdh.PicklesToImageDataset(img_dir='data/images/{}')


def TrainNetwork():
    TrainMobileNet.main()


if __name__ == '__main__':
    # ShowROIs()
    # CaptureAndEditVideo()
    # ExtractTooltips()
    # CreateTrainingData()
    TrainNetwork()
