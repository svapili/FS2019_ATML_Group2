import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
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
from dataAugmenter import preprocessData

import SimpleNet
import train
import test_

import platform


if __name__ == '__main__':

    Linux = False
    Path = ''

    if platform.platform()[0:5] == 'Linux':
        Linux = True
        Path = '/var/tmp/'

    # Directories
if not Linux:
    # Directories
    dataDir = '../data/ISIC-images'
    trainDir = '../data/ISIC-images/train/'
    testDir = '../data/ISIC-images/test/'
    valDir = '../data/ISIC-images/val/'
else:
    dataDir = Path + 'ISIC-images'
    trainDir = Path + 'ISIC-images/train/'
    testDir = Path + 'ISIC-images/test/'
    valDir = Path + 'ISIC-images/val/'


    newDataSplit = False # Set to true to split the data randomly again. Data have first to be downloaded and extracted with data_extractor.py

    dataPreprocessing = False # Set to true to resize and augment the data

    n_epochs = 10

######################
# Splitting the data #
######################
    if (newDataSplit):

        testRatio = .1
        valRatio = .1
        
        split(trainDir, testDir, valDir, testRatio, valRatio)
        
####################################################
# Preprocessing the data (resizing and augmenting) #
####################################################
    if (dataPreprocessing):
        
        preprocessData([trainDir, testDir, valDir])
        
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

    #model
    #model = SimpleNet.ConvNet()
    model = models.AlexNet(num_classes=2)


    model = model.to(device)

    # optimizer definition
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # test train and test function
    #train_loss, train_accuracy = train.train(model, dataloaders['train'], optimizer, loss_fn, device)
    #val_loss, val_accuracy = test_.test(model, dataloaders['val'], loss_fn, device)

    for epoch in range(n_epochs):
        train_loss, train_accuracy = train.train(model, dataloaders['train'], optimizer, loss_fn, device)
        val_loss, val_accuracy = test_.test(model, dataloaders['val'], loss_fn, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
            epoch + 1, n_epochs,
            train_losses[-1],
            train_accuracies[-1],
            val_losses[-1],
            val_accuracies[-1]))
