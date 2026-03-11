# model.py

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1), # i/p channel is 3 (as rgb) and o/p channel is 32 (hyperparameter which gets doubled after every layer)
        nn.ReLU(),
        nn.MaxPool2d(2,2), # Kernel=2, stride=2  (maxpool with these parameter resizes the image size to half i.e now 16)

        nn.Conv2d(32, 64, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(2,2),  # now 8

        nn.Conv2d(64, 128, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(2,2)  # now 4
        )

        self.fc_layers= nn.Sequential(
        nn.Linear(4*4*128, 256), # 4*4*128 is final image size and suppose we take 256 neuron(4x4x128=2048 features which converted into 256)
        nn.ReLU(),

        nn.Linear(256, 10), # and from 256 to 10 classes (there are 10 classes output in CIFAR dataset)
        ) # here we need to apply softmax but as we are using CrossEntropyLoss it already applies Softmax internally.

    def forward(self, x):
        x= self.conv_layers(x)
        x= x.view(x.size(0), -1) # flattening the output
        x= self.fc_layers(x)

        return x