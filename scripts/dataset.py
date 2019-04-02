# Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import os
import json
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MelanomaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_dir, metadata_file, transform=None):
        """
        Args:
            metadata_file(string): Path to the json file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        with open(metadata_file) as f:
            self.metadata = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                                self.metadata[idx]['name'])
        image = plt.imread(img_name + '.jpg')
        classification = self.metadata[idx]['benign_malignant'] # It's unclear to me if we can give a string to the model, or if we should convert each class to a corrosponding int value
        sample = {'image': image, 'classification': classification}

        if self.transform:
            sample = self.transform(sample)

        return sample
