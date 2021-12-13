import torch
from torch import nn
import mss
import yaml
from torchvision import models, transforms
from PIL import Image
import cv2 as cv
import numpy as np


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
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(
        in_features=1024,
        out_features=num_classes,
        bias=True)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    return model


def parse_screen_live(config_dir='screen_cfg.yaml'):
    device = get_device()

    color_filters = [np.full((32, 32, 3), (0, 0, 255), np.uint8),
                     np.full((32, 32, 3), (0, 255, 0), np.uint8)]

    preprocess_trans = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = load_model(num_classes=2, model_dir='models/trained_mobilenet_model.pt')
    model = model.to(device)

    sct = mss.mss()

    config = yaml.safe_load(open(config_dir))

    s_x = config['screen']['x'] + config['regions']['playscreen']['x']
    s_y = config['screen']['y'] + config['regions']['playscreen']['y']
    s_w = config['regions']['playscreen']['w']
    s_h = config['regions']['playscreen']['h']

    while True:
        with torch.no_grad():
            screen = np.array(sct.grab({'top': s_y,
                                        'left': s_x,
                                        'height': s_h,
                                        'width': s_w}), dtype=np.uint8)[:, :, :-1]
            # img = Image.frombytes("RGB", screen.size, screen.bgra, "raw", "BGRX")
            screen_rgb = cv.cvtColor(screen, cv.COLOR_BGR2RGB)
            tiles = [preprocess_trans(
                Image.fromarray(screen_rgb[x:x+32, y:y+32])).float().unsqueeze(0).to(device)
                     for y in range(0, screen.shape[1], 32)
                     for x in range(0, screen.shape[0], 32)]
            # results = model(tiles)
            # print(results)
            # results = list(np.array([model(tile).cpu() for tile in tiles]))
            results = np.array([torch.argmax(model(tile)).cpu()
                                for tile in tiles]).reshape(16, 10)

            for y in range(0, screen.shape[1], 32):
                for x in range(0, screen.shape[0], 32):
                    screen[x:x+32, y:y+32] = cv.addWeighted(
                        screen[x:x+32, y:y+32], 0.8,
                        color_filters[results[y//32,x//32]], 0.2, 0)
            cv.imshow('Labels', screen)
            cv.waitKey(1)
