# How to Train a Classifier

# Setup

1) Put positive images (images containing the object) in the 'positives' directory and negative images (images not containing the object) in the 'negatives' directory.
   - Optionally, put images you would like to use for testing in the 'test' directory.
2) Run `python preprocess_imgs.py`. You must run this any time you change the contents of these directories.
3) Run `. make_annotations.sh` and annotate the positive images. use 'c' to confirm an annotation, 'd' to delete the last annotation, and 'n' to move to the next image.
4) Run `python annotations_to_vec.py`
5) Run `. train_classifier.sh` to train the classifier.

Feel free to modify the parameters in train_classifier.sh

Testing is done by running `python test_detection.py`

# Retraining, Adding Samples

### Retraining

If you want to retrain, you must first delete the contents of the 'classifier' directory.

### Adding more samples:

If you want to add more positive samples:

1) Add new samples to the 'new_positives' directory.
2) Run `python preprocess_imgs.py`.
3) Run `. make_new_annotations.sh`.
4) Append 'new_annotations.txt' to 'annotations.txt' and move the raw and processed images into the 'positives' directory.
5) Run `python annotations_to_vec.py`
6) Run `. train_classifier.sh` to train the classifier.


If you want to add more negative samples:

1) Add new samples to the 'negatives' directory.
2) Run `python preprocess_imgs.py`.
4) Run `python annotations_to_vec.py`
3) Run `. train_classifier.sh` to train the classifier.
