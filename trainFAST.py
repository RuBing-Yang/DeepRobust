from deeprobust.image.defense.pgdtraining import PGDtraining
from deeprobust.image.defense.fast import Fast

from deeprobust.image.config import defense_params
from deeprobust.image.netmodels.CNN import Net
import torch
from torchvision import datasets, transforms 

from torch.utils.data import DataLoader
import setupData

if __name__ == '__main__':
    model = Net()
    train_data = setupData.CustomImageDataset(
        'train_phase1/label.csv', 
        'train_phase1/images/',
        transform = transforms.Compose([
            transforms.Resize((500, 500)),
            #transforms.ToTensor(),
        ]))

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(train_data, batch_size=128, shuffle=True)

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
    defense.generate(train_loader, test_loader, **defense_params["FAST_MNIST"])

