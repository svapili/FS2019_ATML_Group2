import torch
import torchvision
import torchvision.transforms
import numpy as np
import matplotlib.pyplot as plt
from dataset import MelanomaDataset
import util
import glob

# Directories
metadata_path = '../data/metadata.json'
images_path = '../data/ISIC-images/'

if __name__ == '__main__':

    # Dataset loading
    melanoma_dataset = MelanomaDataset(images_path, metadata_path)
    
    # Testing dataset class
    util.showSamples(melanoma_dataset)
    
    
    # TODO: data augmentation for class malignant
    # TODO: split data into train, test and validation
    # TODO: create dataloader
