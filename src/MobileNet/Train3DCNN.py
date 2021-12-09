from src.MobileNet.FramePredDataLoader import FramePredictionDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import cv2 as cv

import copy


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool3d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16)
        )

        self.layer6 = nn.Sequential(
            nn.Upsample(scale_factor=2)
        )

        self.layer7 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.ConvTranspose3d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16)
        )

        self.layer9 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.ConvTranspose3d(8, 1, 3, padding=1),
            # nn.MaxPool3d((6,1,1))
        )

        #self.layer10 = nn.Flatten(start_dim=1)
        self.layer10 = nn.Flatten(start_dim=3)

    def forward(self, x):
        #print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer9(x)
        #print(x.shape)
        x = self.layer10(x)
        x = torch.squeeze(x, dim=1)
        return x


def main():

    num_frames = 3

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name()
        print(f"Running on your {gpu_name} (GPU)")
    else:
        device = torch.device("cpu")
        print("Running on your CPU")

    net = Net().to(device)

    init_input = torch.randn(1, 2, num_frames, 10, 16).to(device)
    out = net(init_input)
    net.zero_grad()
    out.backward(torch.randn(1, 2, 160).to(device))

    train_dataset = FramePredictionDataset('data/videos/cows_train_pred.pkl', device)
    test_dataset = FramePredictionDataset('data/videos/cows_test_pred.pkl', device)
    # train_size = int(0.9 * len(frame_pred_ds))
    # test_size = len(frame_pred_ds) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(frame_pred_ds, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=(5) + 1,
                             shuffle=True, num_workers=0)

    testloader = DataLoader(test_dataset, batch_size=(5) + 1,
                            shuffle=True, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    # optimizer = optim.SGD(net.parameters(), lr=.001)
    optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    num_epochs = 10

    losses = []
    accuracies = []
    val_epochs = []
    val_losses = []
    val_accuracies = []
    epochs = list(x for x in range(num_epochs))

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        if epoch % 1 == 0:
            print('Beginning Epoch {}'.format(epoch))
        running_loss = 0.0
        net.train()
        correct = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['input']
            targets = data['target']
            labels = data['label']
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print('shapes')
            # print(outputs.shape)
            # print(targets.shape)
            loss = criterion(outputs, targets.flatten(start_dim=2))
            loss.backward()
            optimizer.step()

            #print(outputs.shape)
            choice = outputs.argmax(dim=1)
            cur_correct = (choice == labels).float().sum()
            correct += cur_correct
            #print(cur_correct/6)

            # print statistics
            running_loss += loss.item()
        accuracies.append((correct/(len(train_dataset)*160)).cpu())
        losses.append((running_loss/len(train_dataset)))

        if epoch % 1 == 0:
            net.eval()
            val_epochs.append(epoch)
            running_loss = 0.0
            correct = 0.0
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['input']
                targets = data['target']
                labels = data['label']
                # forward + backward + optimize
                outputs = net(inputs)
                choice = outputs.argmax(dim=1)
                correct += (choice == labels).float().sum()
                loss = criterion(outputs, targets.flatten(start_dim=2))
                # print statistics
                running_loss += loss.item()
            val_acc = (correct/(len(test_dataset)*160)).cpu()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(net.state_dict())
            val_accuracies.append(val_acc)
            val_losses.append((running_loss/len(test_dataset)))

    print('Finished Training')
    net.load_state_dict(best_model_wts)
    torch.save(net.state_dict(), 'trained_3dcnn.pt')

    fig, axes = plt.subplots(2, 2)
    axes[0][0].plot(epochs, losses)
    axes[0][0].set_title('Train Loss')
    axes[0][1].plot(val_epochs, val_losses)
    axes[0][1].set_title('Validation Loss')
    axes[1][0].plot(epochs, accuracies)
    axes[1][0].set_title('Train Accuracy')
    axes[1][1].plot(val_epochs, val_accuracies)
    axes[1][1].set_title('Validation Accuracy')

    plt.show()

    color_filters = [np.full((32, 32, 3), (0, 0, 255), np.uint8),
         np.full((32, 32, 3), (245, 66, 114), np.uint8),
         np.full((32, 32, 3), (0, 255, 0), np.uint8),
         np.full((32, 32, 3), (255, 0, 0), np.uint8),
         np.full((32, 32, 3), (255, 255, 0), np.uint8),
         np.full((32, 32, 3), (0, 255, 255), np.uint8),
         np.full((32, 32, 3), (255, 0, 255), np.uint8)]

    baseline = 0
    accuracy = 0
    for data in test_dataset:
        out = torch.argmax(net(data['input'].unsqueeze(0)), dim=1)
        out = out.reshape((10, 16))
        labels = data['label'].reshape((10, 16))

        accuracy += ((out == labels).float().sum())
        last_in = torch.argmax(data['input'].permute(1, 2, 3, 0)[-1], dim=-1)
        baseline += ((last_in == labels).float().sum())

    accuracy = accuracy/(len(test_dataset)*160)
    baseline = baseline/(len(test_dataset)*160)

    print('Total Accuracy: {}% Total Baseline:  {}%'.format(accuracy*100, baseline*100))

    for data in test_dataset:
        out = torch.argmax(net(data['input'].unsqueeze(0).to(device)), dim=1)
        out = out.reshape((10, 16))
        labels = data['label'].reshape((10, 16))

        raw_frame1 = np.zeros((320, 512, 3), dtype=np.uint8)
        for y in range(0, raw_frame1.shape[1], 32):
            for x in range(0, raw_frame1.shape[0], 32):
                raw_frame1[x:x+32, y:y+32] = color_filters[labels[x//32, y//32]]
        cv.imshow('Labels', raw_frame1)
        print(labels)

        raw_frame2 = np.zeros((320, 512, 3), dtype=np.uint8)
        for y in range(0, raw_frame2.shape[1], 32):
            for x in range(0, raw_frame2.shape[0], 32):
                raw_frame2[x:x+32, y:y+32] = color_filters[out[x//32, y//32]]
        cv.imshow('Predictions', raw_frame2)

        raw_frames = np.zeros((num_frames, 320, 512, 3), dtype=np.uint8)
        print(data['input'].shape)
        for idx in range(0, num_frames):
            in_labels = torch.argmax(data['input'].permute(1, 2, 3, 0)[idx], dim=-1)
            for y in range(0, raw_frames[idx].shape[1], 32):
                for x in range(0, raw_frames[idx].shape[0], 32):
                    raw_frames[idx, x:x+32, y:y+32] = color_filters[in_labels[x//32, y//32]]
            cv.imshow('Input {}'.format(idx), raw_frames[idx])

        accuracy = ((out == labels).float().sum())/160
        last_in = torch.argmax(data['input'].permute(1, 2, 3, 0)[2], dim=-1)
        baseline = ((last_in == labels).float().sum())/160

        print('Accuracy: {}% Baseline:  {}%'.format(accuracy, baseline))

        cv.waitKey(0)

        # print('Input: {}'.format(data['input']))
        # print('Label: {}'.format(data['label']))
        # print('Output: {}'.format(out))


def load_and_test():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name()
        print(f"Running on your {gpu_name} (GPU)")
    else:
        device = torch.device("cpu")
        print("Running on your CPU")

    model = Net().to(device)
    model.load_state_dict(torch.load('trained_3dcnn.pt'))
    model.eval()

    test_dataset = FramePredictionDataset('data/videos/cows_test_pred.pkl', device)

    baseline = 0
    accuracy = 0

    for data in test_dataset:
        out = torch.argmax(model(data['input'].unsqueeze(0)), dim=1)
        out = out.reshape((10, 16))
        labels = data['label'].reshape((10, 16))

        accuracy += ((out == labels).float().sum())
        last_in = torch.argmax(data['input'].permute(1, 2, 3, 0)[-1], dim=-1)
        baseline += ((last_in == labels).float().sum())

    accuracy = accuracy/(len(test_dataset)*160)
    baseline = baseline/(len(test_dataset)*160)

    print('Total Accuracy: {}% Total Baseline:  {}%'.format(accuracy*100, baseline*100))
