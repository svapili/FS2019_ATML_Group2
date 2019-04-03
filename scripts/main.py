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

# Directories
data_dir = '../data/ISIC-images/'

def imshow(inp, title=None):
    if title is not None:
        plt.title(title)
    plt.imshow(inp)
    plt.pause(0.001) 

if __name__ == '__main__':
    
    image_dataset = datasets.ImageFolder(data_dir)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4,
                                                 shuffle=True, num_workers=4)
    
    dataset_sizes = len(image_dataset)
    class_names = image_dataset.classes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # NOT WORKING YET - TypeError
#    inputs = next(iter(dataloader))
#    out = torchvision.utils.make_grid(inputs)
#    util.imshow(out)
    
    # TODO: show sample images using dataloader
    # TODO: split data into train, test and validation
    # TODO: data augmentation for class malignant
