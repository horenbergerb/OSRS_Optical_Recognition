from dataclasses import dataclass
import time
import pickle
import os

import mss
import cv2 as cv
import numpy as np
import yaml
import pytesseract
import pyautogui as pag

from src.utils.ExtractionTools import extract_colors


@dataclass
class RawTrainingData:
    img: np.ndarray
    tooltip: np.ndarray
    label: int


def LoadPickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def SavePickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def TooltipsToLabels(samples, labels, config_dir):
    config = yaml.safe_load(open(config_dir))
    keep_colors = config['text_colors']['blue'] + \
                  config['text_colors']['yellow']
    # keep only blue or yellow words and convert to black/white mask
    for idx in range(len(samples)):
        samples[idx].tooltip = extract_colors(
            samples[idx].tooltip,
            keep_colors,
            55)
        # invert since OCR needs black text on white background
        samples[idx].tooltip = cv.cvtColor(
            (~(samples[idx].tooltip)).astype(np.uint8)*255,
            cv.COLOR_GRAY2RGB)
        samples[idx].label = pytesseract.image_to_string(
            samples[idx].tooltip).strip().lower()
        labels.add(samples[idx].label)
    return samples, labels


class TrainingDataGenerator:
    def __init__(self, base_dir='samples_{}.pkl'):
        self.base_dir = base_dir

    # saving and loading is needed since
    # all the collected samples cannot be held in memory at once
    # and we recall them at the end of collection to generate their labels
    def _LoadPickle(self, idx):
        with open(self.base_dir.format(idx), 'rb') as f:
            return pickle.load(f)

    def _SavePickle(self, data, idx):
        with open(self.base_dir.format(idx), 'wb') as f:
            pickle.dump(data, f)
            
    def CollectAndPickleSamples(self,
                                config_dir='screen_cfg.yaml',
                                num_samples=600,
                                batch_size=200,
                                box_l=32,
                                delay=.05,
                                do_print=True):
        '''
        Collecting training data for MobileNet
        Samples 32x32 regions around the mouse along with the current tooltip
        After collection, tooltips are parsed via OCR into text labels
        The results are saved to a set of pickle files
        '''
        config = yaml.safe_load(open(config_dir))

        s_x = config['screen']['x'] + config['regions']['playscreen']['x']
        s_y = config['screen']['y'] + config['regions']['playscreen']['y']
        s_w = config['regions']['playscreen']['w']
        s_h = config['regions']['playscreen']['h']

        sct = mss.mss()
        sample_count = 0
        batches = 0
        labels = set()

        if do_print:
            print('Beginning capture of frame data...')
        while sample_count < num_samples:
            batches += 1
            new_samples = []
            print('Capturing batch {}'.format(batches))
            while len(new_samples) < batch_size:
                # capture mouse and screen
                mouse_pos_i = pag.position()
                screen_i = np.array(
                    sct.grab({'top': s_y,
                              'left': s_x,
                              'height': s_h,
                              'width': s_w}), dtype=np.uint8)
                mouse_pos_f = pag.position()
                screen_f = np.array(
                    sct.grab({'top': s_y,
                              'left': s_x,
                              'height': s_h,
                              'width': s_w}), dtype=np.uint8)

                if (((mouse_pos_i[0]-mouse_pos_f[0])**2) +
                    ((mouse_pos_i[0]-mouse_pos_f[0])**2)) > 8:
                    print('Mouse moving too fast to capture')
                    continue

                screen_i = screen_i[:, :, :-1]
                screen_f = screen_f[:, :, :-1]
                m_x, m_y = mouse_pos_i

                # only sample if the entire box is within the play screen
                if (m_x - box_l > s_x) and \
                   (m_x + box_l < s_x+s_w) and \
                   (m_y - box_l > s_y) and \
                   (m_y + box_l < s_y+s_h):
                    # local mouse coordinates relative to the captured screen
                    m_x_l = m_x - s_x
                    m_y_l = m_y - s_y
                    tooltip = screen_i[5:5+20, 4:4+300]
                    new_samples.append(
                        RawTrainingData(
                            screen_i[m_y_l-(box_l//2):m_y_l+(box_l//2),
                                     m_x_l-(box_l//2):m_x_l+(box_l//2), :],
                            tooltip,
                            -1))
                    sample_count += 1
                time.sleep(delay)
            SavePickle(new_samples,
                       self.base_dir.format(sample_count//batch_size))

        if do_print:
            print('Data capture complete. Parsing tooltips with OCR...')

        for idx in range(1, batches+1):
            samples = LoadPickle(self.base_dir.format(idx))
            samples, labels = TooltipsToLabels(samples,
                                               labels,
                                               config_dir)
            SavePickle(samples, self.base_dir.format(idx))
            SavePickle(labels, self.base_dir.format('labels'))

        if do_print:
            print('Label processing complete.')

    def PicklesToImageDataset(self, img_dir='data/images/{}'):
        '''
        Organizes the labeled data to match the format of
        TorchVision's Image Dataset.
        '''
        label_counts = dict()
        num_files = 0
        while(os.path.exists(self.base_dir.format(num_files + 1))):
            num_files += 1
        print('Number of files detected: {}'.format(num_files))
        for idx in range(1, num_files):
            samples = LoadPickle(self.base_dir.format(idx))
            for sample in samples:
                if sample.label == '':
                    sample.label = 'none'
                if sample.label not in label_counts:
                    label_counts[sample.label] = 1
                    os.mkdir(img_dir.format(sample.label))
                    cv.imwrite(
                        img_dir.format(sample.label) + '/{}.png'.format(1),
                        sample.img)
                else:
                    label_counts[sample.label] += 1
                    cv.imwrite(
                        img_dir.format(sample.label) + '/{}.png'.format(
                            label_counts[sample.label]),
                        sample.img)


if __name__ == '__main__':
    tdh = TrainingDataGenerator(base_dir='data/pickles/samples_{}.pkl')
    tdh.PicklesToImageDataset()
