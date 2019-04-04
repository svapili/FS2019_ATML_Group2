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
        
    # NOT WORKING YET - Dimension error
    #showSample(dataloaders, class_names)
    
    # TODO: show sample images using dataloader
    # TODO: data augmentation for class malignant
