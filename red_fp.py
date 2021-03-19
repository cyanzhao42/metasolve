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
from torchmeta.modules.utils import get_subdict

import csv
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.utils import make_grid, save_image
from utils import *
import lpips
import imageio
import scipy
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
from denoiser.denoiser import DnCNN
from inner_modules import addnoise
import skimage.color
import skimage.metrics
import numpy.fft as fft





def main(mode):
    kernel_size=5
    sigma = 1.5
    torch.cuda.set_device(4)

    l=kernel_size
    sig=sigma
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel_gauss = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel_gauss =  kernel_gauss / np.sum(kernel_gauss)
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()

    PSNRS=[]
    SSIM=[]
    L1=[]
    lpips_loss_alex=[]
    lpips_loss_vgg=[]
    maxiter = 12 #500 for run till converge
    if mode=="convergence":
        maxiter=500
        
    def A_gauss(img):
        [H,W,channel]=img.shape
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        image2 = np.zeros([H,W,channel])
        for c in range(channel):
            image2[:,:,c]= scipy.signal.convolve2d(img[:,:,c],kernel_gauss,mode='same',boundary='wrap')
        return torch.tensor(image2).float().cuda()

    for k in range(500):
        img = '/media/data2/cyanzhao/CelebA/img_align_celeba/img_align_celeba/{}.jpg'.format(162771+k)
        img_H = Image.open(img)
        img_H = img_H.crop((0+20,20+20,178-20,198-20))
        img_H = img_H.resize([64,64],Image.LANCZOS) 
        img_H = (transforms.ToTensor()(img_H).permute(1,2,0).cpu().numpy())
        img_H_torch = torch.tensor(img_H).float().cuda()
        img_L = A_gauss(img_H)
        img_L = (img_L).detach().cpu().numpy()
        H,W,channels = img_H.shape

        if mode=="fixed12" or "convergence":
            model = DnCNN()
            denoiser3 = model.load_from_checkpoint("model_zoo/denoiser/dncnn.ckpt")
        
        #fine tuned denoiser
        if mode=="finetune":
            maxiter = 12
            checkpoint = torch.load("model_zoo/denoiser/redfp_tunedenoiser.pth")
            checkpoint = get_subdict(checkpoint['model_state_dict'],'denoiser')
            denoiser3 = DnCNN()
            denoiser3.load_state_dict(checkpoint)
        
        
        denoiser3.cuda()
        denoiser3.eval()
        denoiser3.freeze()


        psnrs=[]
        objective=[]
        y = torch.tensor((img_L*2)-1).cuda()
        y = addnoise(y,7.65/255)    # add AWGN
        x_est=y*1
        sigma = 7.65/2
        l = 0.02
        mu = 2/(1/(sigma**2)+l)
        torch.no_grad()

        fft_psf = np.zeros([H,W,3])
        t = np.floor(5/2)
        for i in range(3):
            fft_psf[int(H/2+1-t-1):int(H/2+1+t),int(W/2+1-t-1):int(W/2+1+t),i]=kernel_gauss
        fft_psf = fft.fft2(fft.fftshift(fft_psf,axes=(0,1)),axes=(0,1))
        fft_y = fft.fft2(y.squeeze(2).cpu().numpy(),axes=(0,1))
        fft_Ht_y = np.conjugate(fft_psf)*fft_y/sigma**2
        fft_HtH = np.abs(fft_psf)**2/sigma**2

        for i in range(maxiter):
            x_est_u = ((x_est)).unsqueeze(0).permute(0,3,1,2)
            f_x_est = ((denoiser3(x_est_u)).squeeze(0).permute(1,2,0))
            abj1 = 1/2/sigma**2*torch.sum(torch.sum((A_gauss(x_est).cuda().view(-1)-y.view(-1))**2))
            abj2 = torch.sum(torch.sum(l*x_est*(x_est-f_x_est)))    
            objective.append((abj1).detach().cpu().numpy()+(abj2).detach().cpu().numpy())

            b = fft_Ht_y+l*fft.fft2(f_x_est.squeeze(2).detach().cpu().numpy(),axes=(0,1))
            A_matrix = fft_HtH + l
            x_est = np.real(fft.ifft2(b/A_matrix,axes=(0,1)))
            x_est = torch.tensor(x_est).float().cuda()
            psnr_val = skimage.metrics.peak_signal_noise_ratio(((x_est+1)/2).detach().cpu().numpy(),img_H,data_range=1)
            if i>1:
                if psnr_val<psnrs[-1]:
                    break
            psnrs.append(psnr_val)   
        PSNRS.append(max(psnrs))
        print(max(psnrs))
        print(psnrs[-1])
        SSIM.append(skimage.metrics.structural_similarity(((x_est+1)/2).detach().cpu().numpy(), img_H, multichannel=True,data_range=1))
        d = loss_fn_alex( (x_est).unsqueeze(0).permute(0,3,1,2), ((img_H_torch*2)-1).unsqueeze(0).permute(0,3,1,2))
        lpips_loss_alex.append(d.detach().item())
        d = loss_fn_vgg( (x_est).unsqueeze(0).permute(0,3,1,2), ((img_H_torch*2)-1).unsqueeze(0).permute(0,3,1,2))
        lpips_loss_vgg.append(d.detach().item())
        L1.append(((((((x_est+1)/2))-((img_H_torch)))**2).mean()).detach().cpu().numpy())
        # imageio.imwrite("red/trained_{}.png".format(k),(((x_est+1)/2).detach().cpu().numpy()))
    print(sum(PSNRS)/len(PSNRS))
    print(sum(SSIM)/len(SSIM))
    print(sum(lpips_loss_vgg)/len(lpips_loss_vgg))
    print(sum(lpips_loss_alex)/len(lpips_loss_alex))
    print(sum(L1)/len(L1))
    
 
if __name__ == '__main__':
    main("finetune")
