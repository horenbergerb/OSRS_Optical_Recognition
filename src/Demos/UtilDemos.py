import yaml
import cv2 as cv

from src.utils.ScreenManager import ScreenManager
from src.utils.VideoTools import VideoCapturer, VideoEditor
from src.utils.ExtractionTools import extract_colors


def ShowROIs():
    '''Illustrates the different ROIs specified in screen_cfg.yaml'''
    config = yaml.safe_load(open('screen_cfg.yaml'))
    sm = ScreenManager(screen_dims=config['screen'], regions=config['regions'])
    while True:
        sm.update()
        cv.imshow('Annotated Screen', sm.annotated_screen())
        cv.waitKey(10)


def CaptureAndEditVideo():
    '''Allows the user to record and playback a set of frames.
    Useful for sampling colors from the screen and other observations'''
    config = yaml.safe_load(open('screen_cfg.yaml'))
    sm = ScreenManager(screen_dims=config['screen'], regions=config['regions'])
    ah = VideoCapturer(sm)
    frames = ah.capture()
    frames = VideoEditor(frames).playback()


def RecordAndSaveVideo(save_dir):
    '''Allows the user to record vieo and then save the frames
    to a directory'''
    config = yaml.safe_load(open('screen_cfg.yaml'))

    sm = ScreenManager(screen_dims=config['screen'], regions=config['regions'])
    ah = VideoCapturer(sm)
    frames = ah.capture(delay=18)
    for idx, frame in enumerate(frames):
        cv.imwrite(
            save_dir.format(idx),
            frame.img)


def ExtractTooltips():
    '''Demonstrates the ability to extract tooltips
    in real time using color thresholds'''
    config = yaml.safe_load(open('screen_cfg.yaml'))
    sm = ScreenManager(screen_dims=config['screen'], regions=config['regions'])
    while True:
        sm.update()
        for color_name, colors in config['text_colors'].items():
            mask = extract_colors(sm.tooltip, colors, 50)
            cv.imshow('Tooltip', sm.tooltip)
            cv.imshow('{} Mask'.format(color_name), mask.astype(float))
        cv.waitKey(10)
