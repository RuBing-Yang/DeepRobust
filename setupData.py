import os
from numpy import imag
import pandas as pd
from torchvision.io import read_image
from torchvision.io import ImageReadMode

import torch
from torch.utils.data import Dataset
from torchvision import transforms 
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path) #, ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, 1]

        if len(image.shape) < 3 or image.shape[0] != 3:
            # transToRGB = transforms.Compose([transforms.Grayscale(num_output_channels=3)])
            # image = transToRGB(image)
            # image = image.convert('RGB')
            print(image.shape)
            if (image.shape[0] == 1):
                image = torch.cat([image, image, image], dim=0) 
            if (image.shape[0] == 4):
                image = image[0:3]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':
    train_data = CustomImageDataset('train_phase1/label.csv', 'train_phase1/images/')
    print(type(train_data))
    print(len(train_data))
    for i in range(len(train_data)):
        image, label = train_data[i]
        print(i, image.shape, label)
        if len(image.shape) != 3:
            break
        if image.shape[0] != 3:
            break