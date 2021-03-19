from torchvision.utils import make_grid, save_image
import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("{}/pytorch_meta".format(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import random
import time
from collections import OrderedDict
from functools import partial
import scipy.io
import scipy.sparse
import configargparse
import matplotlib.pyplot as plt
from pytorch_prototyping import *
from dataio import *
# from inner_modules import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import lpips
torch.cuda.set_device(4)
batchsize=1
image_resolution = [128,128]
model =Unet(in_channels=3,out_channels=3,nf0=256,num_down=4,max_channels=256,use_dropout=False,outermost_linear=True).cuda()
train_img_dataset = CelebA(
        split='val', datasetSize=5, resolution=image_resolution,subsample_method='smooth',downsampled=True,face=True)
train_coord_dataset = Implicit2DWrapper(
        train_img_dataset, sidelength=image_resolution,oversample=1)

train_dataloader = DataLoader(train_coord_dataset, shuffle=False,batch_size=batchsize, pin_memory=False, num_workers=0)
model_dir = '../model_zoo/inpainting_center/'
A = inpainting_A_siren(image_resolution)
output_dir = "{}results/".format(model_dir)
cond_mkdir(output_dir)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (f'\n\nTraining model with {num_parameters} parameters\n\n')

checkpoint = torch.load("{}UNet.pth".format(model_dir))
checkpoint = checkpoint['model_state_dict']
model.load_state_dict((checkpoint))
all_loss = 0
all_losses = []
model.eval()
PSNRs=[]
SSIMs = []
lpips_loss = []
L1=[]
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
for step, images in enumerate(train_dataloader):
        images=images[1]['img'].reshape(batchsize,128,128,3).cuda()
        inpaintimages = A.unsqueeze(-1).cuda()*images
        output = model(inpaintimages.permute(0,3,1,2))
        loss = ((output.permute(0,2,3,1)-images)**2).mean()
        all_loss +=loss.item()
        if loss.item()>0.01:
                print("here")
        all_losses.append(loss.item())
        model_out = output.permute(0,2,3,1)

        d = loss_fn_alex(lin2img(model_out.reshape(1,-1,3), image_resolution), lin2img(images.reshape(1,-1,3), image_resolution))
        lpips_loss.append(d.detach().item())

        out_img = lin2img(model_out.reshape(1,-1,3), image_resolution).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        out_img += 1
        out_img /= 2.
        out_img = np.clip(out_img, 0., 1.)

        gt_img = lin2img(images.reshape(1,-1,3), image_resolution).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        gt_img += 1
        gt_img /= 2.
        gt_img = np.clip(gt_img, 0., 1.)

        input_imag = lin2img(inpaintimages.reshape(1,-1,3), image_resolution).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        input_imag += 1
        input_imag /= 2.
        input_imag = np.clip(input_imag, 0., 1.)

        L1.append(((gt_img-out_img)**2).mean())

        # if step<16:
        #         imageio.imwrite(os.path.join(output_dir, 'painted{}.png'.format(step)), out_img)
        #         imageio.imwrite(os.path.join(output_dir, 'gt{}.png'.format(step)), gt_img)
        #         imageio.imwrite(os.path.join(output_dir, 'inpainting{}.png'.format(step)), input_imag)
        #         imageio.imwrite(os.path.join(output_dir, 'sum{}.png'.format(step)), np.concatenate([input_imag,out_img,gt_img]))
        PSNR = skimage.measure.compare_psnr(out_img, gt_img, data_range=1)
        SSIM = skimage.metrics.structural_similarity(out_img, gt_img, multichannel=True,data_range=1)
        # PSNR_blur = skimage.measure.compare_psnr(input_imag, gt_img, data_range=1)
        PSNRs.append(PSNR)
        SSIMs.append(SSIM)
        # PSNRs_blur.append(PSNR_blur)


PSNRs.append(sum(PSNRs)/len(PSNRs))
SSIMs.append(sum(SSIMs)/len(SSIMs))
lpips_loss.append(sum(lpips_loss)/len(lpips_loss))
np.save(os.path.join(output_dir, 'lpips_loss' + '_context.npy'), lpips_loss)  
np.save(os.path.join(output_dir, 'PSNR' + '_context.npy'), PSNRs)  
print(sum(L1)/len(L1))
print(sum(PSNRs)/len(PSNRs))
print(sum(lpips_loss)/len(lpips_loss))
print(sum(SSIMs)/len(SSIMs))

