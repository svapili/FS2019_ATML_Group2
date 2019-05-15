import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets


# Create dataset and dataloader objects
def melanomaDataLoader(dataDir, batch_size=32, num_workers=4):
    
    data_transforms = transforms.Compose([
            transforms.Resize(250),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    ])
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(dataDir, x),
                                          data_transforms)
                  for x in ['train', 'test', 'val']}
    
    # Create dataloader objects
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)

                  for x in ['train', 'test', 'val']}
    
    return image_datasets, dataloaders


# Display some sample pictures
def showSample(dataloaders, dataset_sizes, class_names):
    
    plt.figure(figsize = (10,5))
    
    for i in range(0, 10):
                
        # Get random image in training set
        index = np.random.choice(dataset_sizes['train'])
        img =  dataloaders['train'].dataset[index][0]
        
        imgClass =  dataloaders['train'].dataset[index][1]
                
        # Transform image before display        
        inp = img.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
                
        # Display image
        plt.subplot(2,5,i+1)
        plt.imshow(inp)
        plt.title('Class: {}'.format(class_names[imgClass]))
