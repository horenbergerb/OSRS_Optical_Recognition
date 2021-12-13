import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

import pickle


class FramePredictionDataset(Dataset):
    def __init__(self, frames_dir, device, in_frames=3, step=5, out_dist=30):
        self.in_frames = in_frames
        self.out_dist = out_dist
        self.step = step
        self.total_in = self.in_frames*self.step
        self.s = in_frames+out_dist
        with open(frames_dir, 'rb') as f:
            self.frames = torch.stack(pickle.load(f)).float().to(device)
        # self.frames[self.frames > .5] = 1
        # self.frames[self.frames <= .5] = 0
        self.labels = torch.argmax(self.frames, dim=-1).unsqueeze(0).to(device)
        self.frames = self.frames.permute(3, 0, 1, 2)
        self.targets = torch.clone(self.frames)
        self.targets[self.targets > .5] = 1
        self.targets[self.targets <= .5] = 0
        #print(self.targets[:, 1, :, :].permute(1,2,0))
        #print(self.targets[:, 1, :, :].shape)

        print(self.frames.shape)
        print(self.labels.shape)

    def __len__(self):
        return self.frames.shape[1]-(self.total_in + self.out_dist)
        # return (self.frames.shape[1]//(self.s))-1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        '''
        sample = {'input': self.frames[:, idx*self.s:idx*self.s+self.in_frames, :, :],
                  'target': self.frames[:, idx*self.s+self.in_frames+self.out_dist, :, :],
                  'label': self.labels[:, idx*self.s+self.in_frames+self.out_dist, :, :].flatten(start_dim=1).squeeze()}
        '''
        sample = {'input': self.frames[:, idx:idx+self.total_in:self.step, :, :],
                  'target': self.targets[:, idx+self.total_in+self.out_dist, :, :],
                  'label': self.labels[:, idx+self.total_in+self.out_dist, :, :].flatten(start_dim=1).squeeze()}

        return sample
