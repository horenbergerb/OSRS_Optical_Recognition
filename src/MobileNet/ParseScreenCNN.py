import torch
from torch import nn
import mss
import yaml
from torchvision import models, transforms
from PIL import Image
import cv2 as cv
import numpy as np
import time
import os
import re

from src.MobileNet.MobileTrainingData import SavePickle, LoadPickle
from src.MobileNet.TrainCNN import CNNet


def natural_sort(l):
    '''
    I found this on the internet to sort the filenames
    https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name()
        print(f"Running on your {gpu_name} (GPU)")
    else:
        device = torch.device("cpu")
        print("Running on your CPU")
    return device


def load_model(num_classes, model_dir):
    '''Loads a fine-tuned MobileNet'''
    model = CNNet(num_classes)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    return model


def diffuse_over_space(probs):
    x_l = len(probs)
    y_l = len(probs[0])
    for x in range(x_l):
        for y in range(y_l):
            count = 0
            avg = 0.0
            if x > 0:
                avg += probs[x-1][y]
                count += 1
            if x < x_l-1:
                avg += probs[x+1][y]
                count += 1
            if y > 0:
                avg += probs[x][y-1]
                count += 1
            if y < y_l-1:
                avg += probs[x][y+1]
                count += 1
            probs[x][y] = avg/count
    return probs


def diffuse_over_time(probs, prev_probs):
    return torch.add(sum(prev_probs), probs)/(len(prev_probs)+1)


def render_directory(video_dir, probs_dir):
    frames = []

    color_filters = [np.full((32, 32, 3), (0, 0, 255), np.uint8),
                 np.full((32, 32, 3), (245, 66, 114), np.uint8),
                 np.full((32, 32, 3), (0, 255, 0), np.uint8),
                 np.full((32, 32, 3), (255, 0, 0), np.uint8),
                 np.full((32, 32, 3), (255, 255, 0), np.uint8),
                 np.full((32, 32, 3), (0, 255, 255), np.uint8),
                 np.full((32, 32, 3), (255, 0, 255), np.uint8)]

    predictions = LoadPickle(probs_dir)

    frame_files = natural_sort(os.listdir(video_dir))
    for idx, frame_file in enumerate(frame_files):
        raw_frame = cv.imread(video_dir + '/' + frame_file)
        labels = torch.argmax(predictions[idx], dim=-1)

        for y in range(0, raw_frame.shape[1], 32):
            for x in range(0, raw_frame.shape[0], 32):
                raw_frame[x:x+32, y:y+32] = cv.addWeighted(
                    raw_frame[x:x+32, y:y+32], 0.8,
                    color_filters[labels[x//32, y//32]], 0.2, 0)
        frames.append(raw_frame)
    return frames


def predict_for_directory(target_dir,
                          save_dir,
                          num_classes=2,
                          prev_frames=3,
                          diffuse_space=False,
                          diffuse_time=False):
    device = get_device()
    preprocess_trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    model = load_model(num_classes=num_classes,
                       model_dir='trained_cnn_model.pt')
    model = model.to(device)
    softmax = nn.Softmax(dim=1)

    all_predictions = []
    prev_predictions = []

    with torch.no_grad():
        for f_name in natural_sort(os.listdir(target_dir)):
            # https://discuss.pytorch.org/t/slicing-image-into-square-patches/8772/3
            screen_rgb = cv.imread(target_dir + '/' + f_name)
            screen_rgb = cv.cvtColor(screen_rgb, cv.COLOR_BGR2RGB)
            screen_rgb = preprocess_trans(
                Image.fromarray(screen_rgb)).float().to(device)
            tiles = torch.flatten(
                screen_rgb.data.unfold(0, 3, 3).unfold(1, 32, 32).unfold(2, 32, 32),
                start_dim=0,
                end_dim=2)
            predictions = torch.reshape(softmax(model(tiles)).cpu(),
                                        (10, 16, num_classes))
            final_predictions = torch.clone(predictions)
            if diffuse_space:
                final_predictions = diffuse_over_space(final_predictions)
            if diffuse_time and len(prev_predictions) > 0:
                final_predictions = diffuse_over_time(final_predictions, prev_predictions)

            prev_predictions.append(predictions)
            if len(prev_predictions) > prev_frames:
                prev_predictions.pop(0)
            all_predictions.append(final_predictions)
    SavePickle(all_predictions, save_dir)


def parse_screen_live(num_classes,
                      config_dir='screen_cfg.yaml',
                      diffuse_space=False,
                      diffuse_time=False,
                      prev_frames=10):
    device = get_device()

    color_filters = [np.full((32, 32, 3), (0, 0, 255), np.uint8),
                     np.full((32, 32, 3), (245, 66, 114), np.uint8),
                     np.full((32, 32, 3), (0, 255, 0), np.uint8),
                     np.full((32, 32, 3), (255, 0, 0), np.uint8),
                     np.full((32, 32, 3), (255, 255, 0), np.uint8),
                     np.full((32, 32, 3), (0, 255, 255), np.uint8),
                     np.full((32, 32, 3), (255, 0, 255), np.uint8)]

    preprocess_trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    model = load_model(num_classes=num_classes,
                       model_dir='trained_cnn_model.pt')
    model = model.to(device)

    softmax = nn.Softmax(dim=1)

    sct = mss.mss()

    prev_predictions = []

    config = yaml.safe_load(open(config_dir))

    s_x = config['screen']['x'] + config['regions']['playscreen']['x']
    s_y = config['screen']['y'] + config['regions']['playscreen']['y']
    s_w = config['regions']['playscreen']['w']
    s_h = config['regions']['playscreen']['h']

    num_frames = 0
    start = -1
    with torch.no_grad():
        while True:
            if num_frames == 100:
                print('FPS: {}'.format(100/(time.time()-start)))
                num_frames = 0
                start = time.time()

            screen = np.array(sct.grab({'top': s_y,
                                        'left': s_x,
                                        'height': s_h,
                                        'width': s_w}),
                              dtype=np.uint8)[:, :, :-1]
            screen_rgb = cv.cvtColor(screen, cv.COLOR_BGR2RGB)
            screen_rgb = preprocess_trans(
                Image.fromarray(screen_rgb)).float().to(device)
            # https://discuss.pytorch.org/t/slicing-image-into-square-patches/8772/3
            tiles = torch.flatten(
                screen_rgb.data.unfold(0, 3, 3).unfold(1, 32, 32).unfold(2, 32, 32),
                start_dim=0,
                end_dim=2)
            predictions = torch.reshape(softmax(model(tiles)).cpu(),
                                        (10, 16, num_classes))
            final_predictions = torch.clone(predictions)
            if diffuse_space:
                final_predictions = diffuse_over_space(final_predictions)
            if diffuse_time and len(prev_predictions) > 0:
                final_predictions = diffuse_over_time(final_predictions, prev_predictions)

            prev_predictions.append(predictions)
            if len(prev_predictions) > prev_frames:
                prev_predictions.pop(0)
            labels = torch.argmax(final_predictions, dim=-1)

            for y in range(0, screen.shape[1], 32):
                for x in range(0, screen.shape[0], 32):
                    screen[x:x+32, y:y+32] = cv.addWeighted(
                        screen[x:x+32, y:y+32], 0.8,
                        color_filters[labels[x//32, y//32]], 0.2, 0)
            cv.imshow('Labels', screen)
            cv.waitKey(1)
            num_frames += 1
