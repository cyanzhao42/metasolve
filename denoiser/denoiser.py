import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import Dataset
import csv
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.utils import make_grid, save_image
import denoiser.basicblock as B
import numpy as np

# --------------------------------------------
# train denoisers 
# partially credit to  https://github.com/cszn/DPIR
# and  https://github.com/vsitzmann/siren
# --------------------------------------------

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class DnCNN(pl.LightningModule):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=20, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x-n
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = x-self.model(x)
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)
        if batch_idx%10:
            grid = make_grid(torch.cat([x,x_hat,y]),normalize=True) 
            self.logger.experiment.add_image('generated_images_train', grid,self.current_epoch) 
        return loss
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat = x-self.model(x)
        loss = F.mse_loss(x_hat, y)
        self.log('val_loss', loss)
        grid = make_grid(torch.cat([x,x_hat,y]),normalize=True) 
        self.logger.experiment.add_image('generated_images_val', grid,self.current_epoch) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

class IRCNN(pl.LightningModule):
    def __init__(self, in_nc=3, out_nc=3, nc=32):
        super(IRCNN, self).__init__()
        L =[]
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x-n

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = x-self.model(x)
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)
        if batch_idx%10:
            grid = make_grid(torch.cat([x,x_hat,y]),normalize=True) 
            self.logger.experiment.add_image('generated_images_train', grid,self.current_epoch) 
        return loss
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat = x-self.model(x)
        loss = F.mse_loss(x_hat, y)
        self.log('val_loss', loss)
        grid = make_grid(torch.cat([x,x_hat,y]),normalize=True) 
        self.logger.experiment.add_image('generated_images_val', grid,self.current_epoch) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

class CelebA(Dataset):
    def __init__(self, split, datasetSize, resolution=[64,64], subsample_method=None,downsampled=True,face=False,oversample=1):
        # SIZE (178 x 218)
        super().__init__()
        assert split in ['train', 'test', 'val'], "Unknown split"

        self.root = '/media/data2/cyanzhao/CelebA/img_align_celeba/img_align_celeba'
        self.img_channels = 3
        self.fnames = []
        self.datasetSize = datasetSize
        self.image_resolution = resolution
        self.face = face
        self.transform = Compose([
            Resize(resolution[0]),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        with open('/media/data2/cyanzhao/CelebA/list_eval_partition.csv', newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rowreader:
                if split == 'train' and row[1] == '0':
                    self.fnames.append(row[0])
                elif split == 'val' and row[1] == '1':
                    self.fnames.append(row[0])
                elif split == 'test' and row[1] == '2':
                    self.fnames.append(row[0])

        self.downsampled = downsampled
        self.subsample_method=subsample_method
    def __len__(self):
        # return len(self.fnames)
        return self.datasetSize

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx])
        # path = ("/home/cyanzhao/download-1.jpg")
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions

            s = min(width, height)
            if self.face:
                left = (width - s) / 2 + 20
                top = (height - s) / 2 + 20
                right = (width + s) / 2 - 20
                bottom = (height + s) / 2 - 20
            else: 
                left = (width - s) / 2
                top = (height - s) / 2
                right = (width + s) / 2
                bottom = (height + s) / 2
            img = img.crop((left, top, right, bottom))
            if self.subsample_method==None:
                img = img.resize((self.image_resolution[0], self.image_resolution[1]))
            else:
                img = img.resize((self.image_resolution[0], self.image_resolution[1]),Image.LANCZOS)

            img = self.transform(img)
            noise_img = img.clone()
            # noise_level = torch.FloatTensor([np.random.uniform(5, 15)])/255.0
            noise = torch.randn(img.size()).mul_(12.5/255)
            noise_img = noise_img+(noise)

        return noise_img,img

if __name__ == '__main__':
    dir='/media/data2/cyanzhao/meta_solve/denoise/64/resol_64_128epoch_12.5'

    #dataset
    dataset = CelebA(split="train",datasetSize=150000,face=True)
    train_loader = DataLoader(dataset,shuffle=False,
                            batch_size=128, pin_memory=True, num_workers=4)
    dataset = CelebA(split="val",datasetSize=8,face=True)
    val_loader = DataLoader(dataset,shuffle=False,
                            batch_size=8, pin_memory=True, num_workers=4)

    #model
    model = DnCNN()

    #training
    trainer = pl.Trainer(gpus='5',max_epochs=500,track_grad_norm=2,weights_summary="full",weights_save_path='{}checkpoints/'.format(dir))
    trainer.fit(model, (train_loader), (val_loader))
