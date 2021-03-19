import os.path
import cv2
import logging

import numpy as np
from datetime import datetime
from collections import OrderedDict
import hdf5storage
from scipy import ndimage
from tqdm.autonotebook import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import shutil
from torchvision.utils import make_grid, save_image

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_sisr as sr
from utils import utils_image as util
from PIL import Image  
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.abspath('../'))
from denoiser.denoiser import DnCNN
from dataio import *
import utils

import lpips
import skimage.metrics


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()
              

noise_level_img = 7.65/2/255.0         # default: 0, noise level for LR image
noise_level_model = noise_level_img  # noise level of model, default 0
model_name = 'dncnn'           # 'drunet_gray' | 'drunet_color' | 'ircnn_gray' | 'ircnn_color'
testset_name = 'celeba'               # test set,  'set5' | 'srbsd68'
x8 = True                            # default: False, x8 to boost performance
iter_num = 500                 # number of iterations
modelSigma1 = noise_level_model*255.
modelSigma2 = noise_level_model*255.

show_img = False                     # default: False
save_L = True                        # save LR image
save_E = True                        # save estimated image
save_LEH = False                     # save zoomed LR, E and H images
border = 0
# loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
# loss_fn_alex = lpips.LPIPS(net='alex').cuda()
sf = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.cuda.device(4)

# --------------------------------
# load kernel
# --------------------------------
kernels = []
l=5
sig=1.5
ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
xx, yy = np.meshgrid(ax, ax)
kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
kernel =  kernel / np.sum(kernel)
kernels.append(kernel)

rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_model), iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)
            
image_resolution=[64,64]
train_img_dataset = CelebA(
        split='train', datasetSize=150000, resolution=image_resolution,subsample_method='smooth',downsampled=True,face=True)
train_coord_dataset = Implicit2DWrapper(
        train_img_dataset, sidelength=image_resolution,oversample=1)
train_coord_dataset = task_batch(train_coord_dataset,num=3)
train_dataloader = DataLoader(train_coord_dataset, shuffle=False,
                        batch_size=1, pin_memory=True, num_workers=0)

map_location = {'cuda:5':'cuda:{}'.format(torch.cuda.current_device())}
model = DnCNN().cuda()
checkpoint = torch.load("../model_zoo/denoiser/dncnn.ckpt",map_location)
model.load_state_dict(checkpoint["state_dict"])
model.apply(set_bn_eval)
checkpoints_dir="logs/checkpoints/"
summaries_dir = "logs/summary/"
util.mkdir(checkpoints_dir)
writer = SummaryWriter(summaries_dir)
epochs = 1
k = kernel
total_steps = 0
optim = torch.optim.Adam(lr=1e-4, params=model.parameters(), amsgrad=False)
with tqdm(total=len(train_dataloader) * epochs) as pbar:
    train_losses = []
    for epoch in range(epochs):
        for step, images in enumerate(train_dataloader):
            loss_val = 0
            for p in range(3):
                start_time = time.time()
                img_H=images[1][p]['img'].reshape(64,64,3).cuda()
                img_H = (img_H+1)/2
                img_L = ndimage.filters.convolve((img_H).detach().cpu().numpy(), np.expand_dims(k, axis=2), mode='wrap')
                img_L += np.random.normal(0, noise_level_img, img_L.shape) # add AWGN
                img_L_tensor = torch.tensor(img_L).unsqueeze(0).float().cuda()
                k_tensor = torch.tensor(np.expand_dims(kernel, 2)).unsqueeze(0).float().cuda()
                FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor.permute(0,3,1,2), k_tensor.permute(0,3,1,2), sf)
                x = img_L_tensor.permute(0,3,1,2)
                img_H = img_H.unsqueeze(0).permute(0,3,1,2)
                for i in range(12):
                    tau = rhos[i].float().repeat(1, 1, 1, 1)
                    x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)
                    x = model(x*2-1)
                    x = (x+1)/2.
                loss_val += ((x-img_H)**2).mean()
            optim.zero_grad()
            loss_val.backward()
            optim.step()
            pbar.update(1)
            writer.add_scalar("loss",loss_val/3,total_steps)

            if not total_steps % 100:
                tqdm.write("Epoch %d, outer loss %0.6f, iteration time %0.6f" % (epoch, loss_val/3, time.time() - start_time))
                init_vs_updates_allbatch = torch.cat([img_H,img_L_tensor.permute(0,3,1,2),x])
                writer.add_image('train' + 'init_vs_updates_allbatch', make_grid(init_vs_updates_allbatch, nrow=1, scale_each=False, normalize=True), global_step=total_steps)
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':optim.state_dict(),
                        'loss':loss_val}
                        ,os.path.join(checkpoints_dir,"model_current.pth"))
            total_steps +=1
            
