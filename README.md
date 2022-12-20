## Overview
The following repo is a solution to the Squares Dataset problem

## Challenge
The main task is to take various images of different file formats and sizes, and classify them as one of the three labels "a", "b", "c".

## Approach: Multi-layer Perceptron (MLP)
Instead of a regular CNN-based image classification model approach, I went with a multi-layer perceptron model. I ended up with a MLP that contains two hidden layers with 10 neurons each and then an output layer with 3 neurons (one for each label). I used batch-normalization and dropout for regularization, and a soft-max activation layer at the end to get 0-1 probabilities for each class.

The main reason why I went with this approach is due to my intuition that the mono-colored squares have very little spatial information that would be useful for the classification task. They are largely featureless; there's no texture, edges, contour, or other low/high dimensionality features that would help classify a square from being in group A, B, or C. Thus, a CNN wouldn't be all that much more powerful than a MLP that directly takes in the color and size of the square as inputs.

### Dataloader
I implemented a Pytorch dataloader that read in the .bmp, .jpg, and .png files as BGR images. The dataloader then converts each image into an input vector with a size of 5, where the information in the vector contains the BGR color of the square, the length of the square, and length of the image. This also has the added benefit of changing any-sized image into a fixed input size for the model.

### Metric: F1-Score
I leveraged the macro-average F1-score from the Python package torchmetrics. Compared to the accuracy metric, you can get a better understanding of how it performs per-label and it doesn't suffer from data imbalance as much.
The best-performing model got a macro-average F1-Score of 98.7% on the "val" dataset after 25 epochs of training.

