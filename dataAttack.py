from deeprobust.image.attack.pgd import PGD
from deeprobust.image.config import attack_params
from deeprobust.image.utils import download_model
import deeprobust.image.netmodels.resnet as resnet

import torch
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader

import setupData

# URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/CIFAR10_ResNet18_epoch_50.pt"
# download_model(URL, "$MODEL_PATH$")

model = resnet.ResNet18().to('cuda')
model.load_state_dict(torch.load("./models/CIFAR10_ResNet18_epoch_20.pt"))
model.eval()

test_loader  = setupData.CustomImageDataset(
        'train_phase1/label.csv', 
        'train_phase1/images/',
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ConvertImageDtype(torch.float),
            #transforms.Grayscale(num_output_channels=3),
            #transforms.ToTensor(),
        ]))
        
x, y = next(iter(test_loader))
x = x.to('cuda').float()

adversary = PGD(model, 'cuda')
Adv_img = adversary.generate(x, y, **attack_params['PGD_CIFAR10'])