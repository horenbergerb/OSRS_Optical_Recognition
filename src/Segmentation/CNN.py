import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

import time
import copy
import os

from src.utils.TorchUtils import get_device


'''
Much of this code was liberally salvaged from the following tutorial:
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
'''


def make_conv_layer(in_channels, out_channels, kernel=3, batch_norm=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )


class CNNet(nn.Module):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(CNNet, self).__init__()

        self.backbone = nn.Sequential(
            make_conv_layer(3, 8, 3),
            make_conv_layer(8, 16, 3),
            nn.MaxPool2d(2),
            make_conv_layer(16, 16, 3),
            make_conv_layer(16, 16, 3)
        )

        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1600, 128),
            nn.Linear(128, 64),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def get_dataloaders(data_dir='./data/images', input_size=32, batch_size=16):
    '''Create training and validation datasets
    Using this resource:
    https://stackoverflow.com/questions/51782021/how-to-use-different-data-augmentation-for-subsets-in-pytorch
    '''

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ]),
    }

    image_dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(image_dataset))
    test_size = len(image_dataset) - train_size

    train, val = torch.utils.data.random_split(
        image_dataset, [train_size, test_size])

    train.dataset = copy.copy(image_dataset)
    train.dataset.transform = data_transforms['train']
    val.dataset.transform = data_transforms['val']

    image_datasets = {'train': train, 'val': val}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4) for x in ['train', 'val']}
    return dataloaders_dict


def train_model(model, dataloaders, criterion,
                optimizer, scheduler, num_epochs=25):
    '''The actual training procedure for a CNN classifier'''

    device = get_device()

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def main():
    '''Creates and trains a CNN to classify images
    based on the data/images directory.
    Saves the CNN to 'models/trained_cnn_model.pt' '''

    # using ImageFolder dataset via torchvision
    data_dir = "./data/images"
    batch_size = 16
    num_epochs = 15
    # assumes 32x32 images
    input_size = 32

    num_classes = len(os.listdir(data_dir))

    model = CNNet(num_classes)

    print("Initializing Datasets and Dataloaders...")
    dataloaders_dict = get_dataloaders(data_dir=data_dir,
                                       input_size=input_size,
                                       batch_size=batch_size)

    # Detect if we have a GPU available
    device = get_device()

    # Send the model to GPU
    model = model.to(device)

    optimizer_ft = optim.AdamW(model.parameters(), lr=.001)
    scheduler_ft = StepLR(optimizer_ft, step_size=5, gamma=0.01)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model, hist = train_model(model,
                              dataloaders_dict,
                              criterion,
                              optimizer_ft,
                              scheduler_ft,
                              num_epochs=num_epochs)

    torch.save(model.state_dict(), 'models/trained_cnn_model.pt')

    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = []

    ohist = [h.cpu().numpy() for h in hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs+1), ohist)
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.show()
