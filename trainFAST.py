from deeprobust.image.defense.fast import Fast
from deeprobust.image.config import defense_params

import resnet
# import deeprobust.image.netmodels.resnet as resnet

import torch
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader

import setupData

if __name__ == '__main__':

    clean_train_data = setupData.CustomImageDataset(
        'train_phase1/label.csv', 
        'train_phase1/images/',
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ConvertImageDtype(torch.float),
            #transforms.Grayscale(num_output_channels=3),
            #transforms.ToTensor(),
        ]))#

    train_loader = DataLoader(clean_train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(clean_train_data, batch_size=4, shuffle=True)

    # resnet50
    train_net = resnet.Net(resnet.Bottleneck, [3,8,36,3], num_classes=20)
    
    defense = Fast(train_net, 'cuda')
    defense.generate(train_loader, test_loader) #, **defense_params["FAST_MNIST"])

