#import torchvision.datasets as datasets
from torchvision import datasets, transforms

from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
        
        
class MelanomaDataset(datasets.ImageFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        resize_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a resized version. E.g., ``transforms.Resize``
        class1_augmentation(callable, optional): A function/transform that takes in a PIL
            image and does data augmentation on the data belonging to class with index 1. 
            E.g. ``transform.RandomCrop``
        tensor_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version of type torch.Tensor. E.g, ``transforms.ToTensor``
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, 
                 resize_transform=None, 
                 class1_augmentation = None,
                 tensor_transform=None,
                 loader=default_loader):
        super(MelanomaDataset, self).__init__(root, transform=None,
                                          target_transform=None,
                                          loader=default_loader)
        self.resize_transform = resize_transform
        self.class1_augmentation = class1_augmentation
        self.tensor_transform = tensor_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.resize_transform is not None:
            sample = self.resize_transform(sample)
        if self.class1_augmentation is not None:
            if target == 1: # '1' is malignant, '0' is benign
                sample = self.class1_augmentation(sample)
        if self.tensor_transform is not None:
            sample = self.tensor_transform(sample)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        

        return sample, target
        
    