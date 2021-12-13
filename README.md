# Overview

This began as a collection of general tools which can be used to parse a live feed of a video game into into actionable intelligence. I was using Old School Runescape as the subject of testing.

This project is now focused on implementing real-time image segmentation to parse the play screen of OSRS. It will be trained on easily-obtainable training data.

The project is divided into three broad components

![](docs/OverviewFlowchart.png)

# Running the Code

Browse the `demos.py` file for a survey of the various implemented features.

# Implementation Details

## Data Collection/Preprocessing

First we collect triplets of (Mouse Position, Tooltip, 32x32 Screenshot centered on mouse)

![](docs/DataCollectionLoop.png)

Next we generate text labels for these triplets using OCR on the Tooltip. We also reformat the labeled data to be compatible with Torchvision's ImageFolder dataset

![](docs/LabelingImageDatasetLoop.png)

## Machine Learning Training

Once the training data is in the format of an ImageFolder dataset, it's pretty cut and dry to load up a neural network and train using Pytorch.

## Real-time Screen Segmentation

The segmentation operation proceeds as follows:

1) Screenshot the play screen
2) Divide playscreen into a grid of 32x32 regions
3) Classify each region
4) Display color-coded classifications

There are postprocessing methods that need documentation

## Future Frame Prediction

This is a new feature which needs documentation
