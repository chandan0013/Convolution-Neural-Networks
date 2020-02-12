# Convolution-Neural-Networks-CNNs

Convolutional Neural Networks (CNNs) for solving image classification task.

will train a CNN on Fashion MNIST data. The network architecture contains 4 CNN layers
followed by one pooling layer and a final fully connected layer. The basic architecture (in
sequential order) will be as follows:

First CNN layer: input channels - 1, output channels - 8, kernel size = 5, padding = 2, stride
= 2 followed by ReLU operation

Second CNN layer: input channels - 8, output channels - 16, kernel size = 3, padding = 1,
stride = 2 followed by ReLU operation

Third CNN layer: input channels - 16, output channels - 32, kernel size = 3, padding = 1,
stride = 2 followed by ReLU operation

Fourth CNN layer: input channels - 32, output channels - 32, kernel size = 3, padding = 1,
stride = 2 followed by ReLU operation

one \Average" pooling layer (nn.AdaptiveAvgPool2d(1) would work in PyTorch)

Plot the training and testing accuracy as a function of atleast 10 epochs.
