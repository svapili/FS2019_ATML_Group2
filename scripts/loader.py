import os
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from dataset import MelanomaDataset


# Create dataset and dataloader objects
def melanomaDataLoader(dataDir):
    
    #TODO: normalize data    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((300,300)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((300,300)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((300,300)),
            transforms.ToTensor(),
        ]),
    }
    
    # TODO: add more augmentation transforms
    malignant_augmentation = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0)
        ]),
        'test': transforms.Compose([
            
        ]),
        'val': transforms.Compose([
          
        ]),
    }
    
    image_datasets = {x: MelanomaDataset(os.path.join(dataDir, x),
                                          data_transforms[x],
                                          malignant_augmentation[x])

                  for x in ['train', 'test', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)

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
        #img = img.numpy().transpose(1,2,0)
        
        # Display image
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        plt.title('Class: {}'.format(class_names[imgClass]))