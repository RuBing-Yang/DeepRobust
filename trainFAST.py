from deeprobust.image.defense.pgdtraining import PGDtraining
from deeprobust.image.defense.fast import Fast

from deeprobust.image.config import defense_params
from deeprobust.image.netmodels.resnet import Net
import torch
from torchvision import datasets, transforms 

from torch.utils.data import DataLoader
import setupData

if __name__ == '__main__':
    model = Net(block=1000, num_blocks=16, num_classes=20)
    train_data = setupData.CustomImageDataset(
        'train_phase1/label.csv', 
        'train_phase1/images/',
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float),
            transforms.Grayscale(num_output_channels=3),
            #transforms.ToTensor(),
        ]))

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    # train_loader = torch.utils.data.DataLoader(
    #                 datasets.MNIST('deeprobust/image/defense/data', train=True, download=True,
    #                                 transform=transforms.Compose([transforms.ToTensor()])),
    #                                 batch_size=100,shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #             datasets.MNIST('deeprobust/image/defense/data', train=False,
    #                             transform=transforms.Compose([transforms.ToTensor()])),
    #                             batch_size=1000,shuffle=True)

    # defense = PGDtraining(model, 'cuda')
    # defense.generate(train_loader, test_loader, **defense_params["PGDtraining_MNIST"])
    defense = Fast(model, 'cuda')
    defense.generate(train_loader, test_loader) #, **defense_params["FAST_MNIST"])

