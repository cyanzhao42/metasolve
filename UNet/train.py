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

torch.cuda.set_device(6)
batchsize=64
image_resolution = [32,32]
model =Unet(in_channels=3,out_channels=3,nf0=256,num_down=4,max_channels=256,use_dropout=False,outermost_linear=True).cuda()
train_img_dataset = CelebA(
        split='train', datasetSize=160000, resolution=image_resolution,subsample_method='smooth',downsampled=True,face=False)
train_coord_dataset = Implicit2DWrapper(
        train_img_dataset, sidelength=image_resolution,oversample=1)
train_coord_dataset = ImageGeneralizationWrapper(train_coord_dataset,generalization_mode = 'conv_cnp',train_sparsity_range=(10, 1000))

train_dataloader = DataLoader(train_coord_dataset, shuffle=True,batch_size=batchsize, pin_memory=True, num_workers=4)

val_img_dataset = CelebA(
        split='val', datasetSize=32, resolution=image_resolution,subsample_method='smooth',downsampled=True,face=False)
val_coord_dataset = Implicit2DWrapper(
        val_img_dataset, sidelength=image_resolution,oversample=1)
val_coord_dataset = ImageGeneralizationWrapper(val_coord_dataset,generalization_mode = 'conv_cnp',train_sparsity_range=(10, 1000))

val_dataloader = DataLoader(val_coord_dataset, shuffle=True,batch_size=32, pin_memory=True, num_workers=4)

model_dir = 'path_to_store_model/'.format(batchsize)
A = inpainting_A_siren(image_resolution)


optim = torch.optim.Adam(lr=1e-5, params=model.parameters(), amsgrad=False)

summaries_dir = os.path.join(model_dir, 'summaries')
cond_mkdir(summaries_dir)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')
cond_mkdir(checkpoints_dir)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (f'\n\nTraining model with {num_parameters} parameters\n\n')

writer = SummaryWriter(summaries_dir)
epochs = 100

total_steps = 0
with tqdm(total=len(train_dataloader) * epochs) as pbar:
    train_losses = []
    for epoch in range(epochs):
        for step, image_load in enumerate(train_dataloader):
                
            start_time = time.time()
            images=image_load[1]['img'].reshape(batchsize,32,32,3).cuda()
            inpaintimages = image_load[0]['img_sparse'].permute(0,2,3,1).cuda()
            output = model(inpaintimages.permute(0,3,1,2))
            loss = ((output.permute(0,2,3,1)-images)**2).mean()
            writer.add_scalar("train_loss",loss,total_steps)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.update(1)

            
            if not total_steps % 1000:
                tqdm.write("Epoch %d, outer loss %0.6f, iteration time %0.6f" % (epoch, loss, time.time() - start_time))
                summary_fn_unet(gt=images.reshape(batchsize,-1,3), observation=inpaintimages.reshape(batchsize,-1,3), model_outputs=output.permute(0,2,3,1).reshape(batchsize,-1,3), writer=writer,total_steps=total_steps,prefix="train",image_resolution=image_resolution)

                for name, parameter in model.named_parameters():
                        writer.add_histogram("train" + name, parameter.cpu(), global_step=total_steps)
                        writer.add_histogram("train" + name + "grad", parameter.grad.cpu(), global_step=total_steps)
                

                torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':optim.state_dict(),
                                'loss':loss}
                                ,os.path.join(checkpoints_dir,"model_current.pth"))
                
                model.eval()
                for step, images_load in enumerate(val_dataloader):
                        images=images_load[1]['img'].reshape(32,32,32,3).cuda()
                        inpaintimages = images_load[0]['img_sparse'].permute(0,2,3,1).cuda()
                        output = model(inpaintimages.permute(0,3,1,2))
                        loss = ((output.permute(0,2,3,1)-images)**2).mean()
                        writer.add_scalar("val_loss",loss,total_steps)
                        summary_fn_unet(gt=images.reshape(32,-1,3), observation=inpaintimages.reshape(32,-1,3), model_outputs=output.permute(0,2,3,1).reshape(32,-1,3), writer=writer,total_steps=total_steps,prefix="val",image_resolution=image_resolution)
                print("eval:{}".format(loss.detach()))
                model.train()

            total_steps = total_steps+1

            
                
