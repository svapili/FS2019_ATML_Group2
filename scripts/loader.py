import os
import torch
import torchvision
from torchvision import datasets, transforms

def melanomaDataLoader(dataDir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(dataDir, x),
                                          data_transforms[x])
                  for x in ['train', 'test', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'test', 'val']}
    
    return image_datasets, dataloaders

def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
#        mean = np.array([0.485, 0.456, 0.406])
#        std = np.array([0.229, 0.224, 0.225])
#        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

def showSample(dataloaders, class_names):

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    
    imshow(out, title=[class_names[x] for x in classes])