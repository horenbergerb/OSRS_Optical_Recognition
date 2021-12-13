import torch
from torch import nn
import mss
import yaml
from torchvision import transforms
from PIL import Image
import cv2 as cv
import numpy as np
import time
import os

from src.utils.PickleUtils import SavePickle, LoadPickle
from src.utils.FileUtils import natural_sort
from src.utils.TorchUtils import get_device
from src.Segmentation.CNN import CNNet


color_filters = [np.full((32, 32, 3), (0, 0, 255), np.uint8),
                 np.full((32, 32, 3), (245, 66, 114), np.uint8),
                 np.full((32, 32, 3), (0, 255, 0), np.uint8),
                 np.full((32, 32, 3), (255, 0, 0), np.uint8),
                 np.full((32, 32, 3), (255, 255, 0), np.uint8),
                 np.full((32, 32, 3), (0, 255, 255), np.uint8),
                 np.full((32, 32, 3), (255, 0, 255), np.uint8)]


def load_model_for_prediction(num_classes, model_dir):
    '''Loads a fine-tuned MobileNet'''
    model = CNNet(num_classes)
    model.load_state_dict(torch.load(model_dir))
    model.head = nn.Sequential(
        *model.head,
        nn.Softmax(dim=1)
    )
    model.eval()

    return model


def diffuse_over_space(probs):
    '''Iteratively sets each tile's probability
    equal to the average of its neighbors.'''
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
    '''Sets each tile's probability equal to the average of the
    tile's probabilities over the current and preceeding frames'''
    return torch.add(sum(prev_probs), probs)/(len(prev_probs)+1)


def predict_frame(frame, model, device, prev_probs,
                  diffuse_space=False, diffuse_time=False):
    '''Computes the probabilites of each classification for every tile
    in the frame. These probabilities describe the segmentation'''
    preprocess_trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    frame = preprocess_trans(
        Image.fromarray(frame)).float().to(device)
    tiles = torch.flatten(
        frame.data.unfold(0, 3, 3).unfold(1, 32, 32).unfold(2, 32, 32),
        start_dim=0,
        end_dim=2)
    predictions = torch.reshape(model(tiles).cpu(),
                                (10, 16, model.num_classes))
    final_predictions = torch.clone(predictions)
    if diffuse_space:
        final_predictions = diffuse_over_space(final_predictions)
    if diffuse_time and len(prev_probs) > 0:
        final_predictions = diffuse_over_time(final_predictions, prev_probs)

    return final_predictions


def render_frame(frame, probs):
    '''Applies color filters to the frame in order to
    visually demonstrate the segmentation'''
    global color_filters

    labels = torch.argmax(probs, dim=-1)
    for y in range(0, frame.shape[1], 32):
        for x in range(0, frame.shape[0], 32):
            frame[x:x+32, y:y+32] = cv.addWeighted(
                frame[x:x+32, y:y+32], 0.8,
                color_filters[labels[x//32, y//32]], 0.2, 0)
    return frame


def predict_directory(target_dir,
                      save_dir,
                      num_classes=2,
                      prev_frames=3,
                      diffuse_space=False,
                      diffuse_time=False):
    '''Given a directory, this function predicts the segmentation
    probabilities for every frame and saves the predictions to
    a pickle file. This can then be used for further processing,
    such as the render_directory function above'''
    device = get_device()
    model = load_model_for_prediction(num_classes=num_classes,
                                      model_dir='models/trained_cnn_model.pt')
    model = model.to(device)

    all_predictions = []
    prev_predictions = []

    with torch.no_grad():
        for f_name in natural_sort(os.listdir(target_dir)):
            # https://discuss.pytorch.org/t/slicing-image-into-square-patches/8772/3
            screen_rgb = cv.imread(target_dir + '/' + f_name)
            screen_rgb = cv.cvtColor(screen_rgb, cv.COLOR_BGR2RGB)
            predictions = predict_frame(screen_rgb, model,
                                        device, prev_predictions,
                                        diffuse_space, diffuse_time)
            prev_predictions.append(predictions)
            if len(prev_predictions) > prev_frames:
                prev_predictions.pop(0)
            all_predictions.append(predictions)
    SavePickle(all_predictions, save_dir)


def render_directory(video_dir, probs_dir):
    '''Given a directory of images and corresponding segmentation probabilities
    (in a pickle file), this function will apply color filters to the images
    in order to visually demonstrate the segmentation'''

    frames = []

    predictions = LoadPickle(probs_dir)

    frame_files = natural_sort(os.listdir(video_dir))
    for idx, frame_file in enumerate(frame_files):
        raw_frame = cv.imread(video_dir + '/' + frame_file)
        raw_frame = render_frame(raw_frame, predictions[idx])
        frames.append(raw_frame)
    return frames


def segment_screen_live(num_classes,
                        config_dir='screen_cfg.yaml',
                        diffuse_space=False,
                        diffuse_time=False,
                        prev_frames=10):
    '''Performs segmentation in real time and
    displays the results.'''
    device = get_device()

    model = load_model_for_prediction(num_classes=num_classes,
                                      model_dir='models/trained_cnn_model.pt')
    model = model.to(device)

    sct = mss.mss()

    prev_predictions = []

    config = yaml.safe_load(open(config_dir))

    s = config['screen']
    p = config['regions']['playscreen']
    s_x = s['x'] + p['x']
    s_y = s['y'] + p['y']
    s_w = p['w']
    s_h = p['h']

    num_frames = 0
    start = time.time()
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
            predictions = predict_frame(screen_rgb, model,
                                        device, prev_predictions,
                                        diffuse_space, diffuse_time)

            prev_predictions.append(predictions)
            if len(prev_predictions) > prev_frames:
                prev_predictions.pop(0)

            screen = render_frame(screen, predictions)

            cv.imshow('Labels', screen)
            cv.waitKey(1)
            num_frames += 1
