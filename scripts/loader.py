import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from dataset import MelanomaDataset


# Create dataset and dataloader objects
def melanomaDataLoader(dataDir):
    
    # Define image size
    data_resize_transform = transforms.Resize((300,300))
    
    # Define augmentation transform for the malignant data
    malignant_data_augmentation = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0)
            # TODO: add more transforms
        ]),
        'test': transforms.Compose([
            
        ]),
        'val': transforms.Compose([
          
        ]),
    }
    
    # Define transforms on tensor type
    data_tensor_transform = transforms.Compose([
            transforms.ToTensor()
            # TODO: Normalize data
        ])
    
    # Create dataset objects
    image_datasets = {x: MelanomaDataset(os.path.join(dataDir, x),
                                         data_resize_transform,
                                         malignant_data_augmentation[x],
                                         data_tensor_transform)
    
                  for x in ['train', 'test', 'val']}
    
    # Create dataloader objects
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
        img = img.numpy().transpose(1,2,0)
        
        # Display image
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        plt.title('Class: {}'.format(class_names[imgClass]))