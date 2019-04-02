import torch
import torchvision
import torchvision.transforms
import numpy as np
import matplotlib.pyplot as plt
from dataset import MelanomaDataset

# Directories
metadata_path = '../data/metadata.json'
images_path = '../data/ISIC-images/'

# Dataset loading
melanoma_dataset = MelanomaDataset(images_path, metadata_path)

# Testing dataset class
sample_idx = 20
fig = plt.figure()
plt.title(melanoma_dataset[sample_idx]['classification'])
plt.imshow(melanoma_dataset[sample_idx]['image'])
