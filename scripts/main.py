import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import util
from dataSplitter import split
from loader import melanomaDataLoader, showSample

import SimpleNet
import train
import test_


if __name__ == '__main__':
    
    # Directories
    dataDir = '../data/ISIC-images'

    newDataSplit = False # Set to true to split the data randomly again. Data have first to be downloaded and extracted with data_extractor.py

######################
# Splitting the data #
######################
    if (newDataSplit):
        
        trainDir = '../data/ISIC-images/train/'
        testDir = '../data/ISIC-images/test/'
        valDir = '../data/ISIC-images/val/'

        testRatio = .1
        valRatio = .1
        
        split(trainDir, testDir, valDir, testRatio, valRatio)
        
##############################
# Creating DataLoader object #
##############################
# Based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    
    image_datasets, dataloaders = melanomaDataLoader(dataDir)
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'val']}
    class_names = image_datasets['train'].classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    showSample(dataloaders, dataset_sizes, class_names)
    
##############################
# Configuring Network        #
##############################

    # Learning rate config
    learning_rate = 0.001

    model = SimpleNet.ConvNet()
    model = model.to(device)

    # optimizer definition
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # test train and test function
    train_loss, train_accuracy = train.train(model, dataloaders['train'], optimizer, loss_fn, device)
    val_loss, val_accuracy = test_.test(model, dataloaders['val'], loss_fn, device)