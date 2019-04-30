import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets


# Create dataset and dataloader objects
def melanomaDataLoader(dataDir):
    
    data_transforms = transforms.Compose([
            transforms.ToTensor()
            ])
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(dataDir, x),
                                          data_transforms)
                  for x in ['train', 'test', 'val']}
    
    # Create dataloader objects
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                 shuffle=True, num_workers=2)

                  for x in ['train', 'test', 'val']}
    
    return image_datasets, dataloaders


# Display some sample pictures
def showSample(dataloaders, dataset_sizes, class_names):
    
    plt.figure()
    
    for i in range(0, 10):
        
        # Get random image in training set
        index = np.random.choice(dataset_sizes['train'])
        img =  dataloaders['train'].dataset[index][0]
        imgClass =  dataloaders['train'].dataset[index][1]
                
        # Transform image before display
        img = img.numpy().transpose(1,2,0)
        
        # Display image
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        plt.title('Class: {}'.format(class_names[imgClass]))