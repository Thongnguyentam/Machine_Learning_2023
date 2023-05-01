# https://nextjournal.com/gkoehler/pytorch-mnist
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
#criterion is a PyTorch loss function that computes the loss between the predicted output 
criterion = nn.CrossEntropyLoss()

#Define hyperparemeters
batch_size = 32
batch_size_test = 10
learning_rate = 0.01
num_epochs = 20
input_size = 784  #28x28 pixels
hidden_size1 = 300
hidden_size2 = 200
num_classes = 10