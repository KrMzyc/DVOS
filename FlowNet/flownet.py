# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:11
Modified for PyTorch conversion by ChatGPT
"""

# Import Libraries:
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

class FlowNetS(nn.Module):
    def __init__(self, height, width):
        super(FlowNetS, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.leaky_relu_1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.leaky_relu_2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.leaky_relu_3 = nn.LeakyReLU(0.1)

        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.leaky_relu_3_1 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.leaky_relu_4 = nn.LeakyReLU(0.1)

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.leaky_relu_4_1 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.leaky_relu_5 = nn.LeakyReLU(0.1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.leaky_relu_5_1 = nn.LeakyReLU(0.1)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.leaky_relu_6 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu_1(self.conv1(x))
        x = self.leaky_relu_2(self.conv2(x))
        x = self.leaky_relu_3(self.conv3(x))
        x = self.leaky_relu_3_1(self.conv3_1(x))
        x = self.leaky_relu_4(self.conv4(x))
        x = self.leaky_relu_4_1(self.conv4_1(x))
        x = self.leaky_relu_5(self.conv5(x))
        x = self.leaky_relu_5_1(self.conv5_1(x))
        x = self.leaky_relu_6(self.conv6(x))
        return x

def load_weights(model, checkpoint_path):
    # Load the checkpoint that contains the state_dict
    checkpoint = torch.load(checkpoint_path)
    
    # Extract the state_dict which contains the model's weights
    state_dict = checkpoint['state_dict']

    # Load the state_dict into the model
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully from checkpoint.")


