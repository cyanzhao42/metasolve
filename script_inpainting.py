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
from dataio import *
from functools import partial
from modules import *
import scipy.io
import scipy.sparse
import configargparse
import matplotlib.pyplot as plt
from utils import *
from meta_modules import *
from inner_modules import *
from train import *

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False,
      is_config_file=True, help='Path to config file.')

p.add_argument('--note',type=str,default='omage90_g0.9_l1')
p.add_argument('--omega0',type=int,default=90)
p.add_argument('--task',type=str,default='deblur')
p.add_argument('--grey',type=bool, default=True)
p.add_argument('--gradient_type',type=str,default="static")
p.add_argument('--num_hidden_layers',type=int,default=3)
p.add_argument('--oversample',type=int, default=1)
p.add_argument('--hidden_features',type=int,default=256)
p.add_argument('--resolution',type=int,default=128)
p.add_argument('--kernel_size',type=int,default=5)
p.add_argument('--sigma',type=float,default=1.5)
p.add_argument('--add_noise',type=float,default=0)
p.add_argument('--lr',type=float,default=1e-5)
p.add_argument('--inner_lr_init',type=float,default=1e-2)
p.add_argument('--lr_type',type=str,default="per_step")
p.add_argument('--num_meta_steps',type=int,default=3)
p.add_argument('--class_solver',type=str,default="gd")
p.add_argument('--gradient_steps',type=float,default=0.9)
p.add_argument('--num_updates',type=int,default=1)
p.add_argument('--datasetSize',type=int,default=150000)
p.add_argument('--subsample',type=str,default='smooth')
p.add_argument('--face',type=bool,default=True)
p.add_argument('--batch_size',type=int,default=5)
p.add_argument('--epochs',type=int,default=3)
p.add_argument('--steps_til_summary',type=int,default=100)
p.add_argument('--epochs_til_checkpoint',type=int,default=10)
p.add_argument('--logging_root', type=str,
               default='/media/data/test/inpainting/', help='root for logging')
p.add_argument('--cuda',type=int,default=4)
opt = p.parse_args()

model_dir = "{}resol{}_face{}_csolver{}_ks{}_sig{}_noise{}_lr{}_innerlr{}_innerlrtyp{}_nummeta{}_numsolver{}_solverlr{}_data{}_sample{}_batch{}_note{}_g{}_hf{}/"\
    .format(opt.logging_root,opt.resolution,opt.face,opt.class_solver,opt.kernel_size,opt.sigma,opt.add_noise,opt.lr,opt.inner_lr_init,opt.lr_type,\
    opt.num_meta_steps,opt.num_updates,opt.gradient_steps,opt.datasetSize,opt.subsample,opt.batch_size,opt.note,opt.gradient_type,opt.hidden_features)
print(model_dir)
if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == "y":
            shutil.rmtree(model_dir)
os.makedirs(model_dir)

##copy files
os.makedirs(model_dir + '/codes')
shutil.copy('script_inpainting.py', model_dir + 'codes/script_inpainting.py')
shutil.copy('dataio.py', model_dir + 'codes/dataio.py')
shutil.copy('inner_modules.py', model_dir + 'codes/inner_modules.py')
shutil.copy('meta_modules.py', model_dir + 'codes/meta_modules.py')
shutil.copy('modules.py', model_dir + 'codes/modules.py')
shutil.copy('train.py', model_dir + 'codes/train.py')
shutil.copy('utils.py', model_dir + 'codes/utils.py')


torch.cuda.set_device(opt.cuda)
torch.manual_seed(0)

siren = SingleBVPNet(in_features=2, num_hidden_layers=opt.num_hidden_layers,hidden_features=opt.hidden_features, out_features=3,omega0=opt.omega0).cuda()
image_resolution = [opt.resolution,opt.resolution]
A = inpainting_A_siren(image_resolution)
inner_loss = partial(residual_inpainting_siren,A=A)
classical_solver = partial(gradient_decent,A=A)

model = meta_solver(siren, A, inner_loss,opt.inner_lr_init,classical_solver,opt.num_updates,\
    opt.gradient_steps,num_meta_steps=opt.num_meta_steps,lr_type=opt.lr_type,task=opt.task,\
    gradient_type=opt.gradient_type,grey=opt.grey,mode="train",oversample=opt.oversample,\
    gradient_limit=None,resolution=opt.resolution,final_gradient_steps=1)
model.cuda()
outer_loss = partial(lploss,p=1)
summary_fn = partial(summary_fn,image_resolution=image_resolution)

train_img_dataset = CelebA(
        split='train', datasetSize=opt.datasetSize, resolution=image_resolution,subsample_method=opt.subsample,downsampled=True,face=opt.face)
train_coord_dataset = Implicit2DWrapper(
        train_img_dataset, sidelength=image_resolution,oversample=opt.oversample)
train_generalization_dataset = ImageGeneralizationWrapper(train_coord_dataset, test_sparsity='empty',train_sparsity_range=(90,100),generalization_mode='conv_cnp')
train_dataset_batch = task_batch(train_generalization_dataset,num=opt.batch_size)
train_dataloader = DataLoader(train_dataset_batch, shuffle=True,batch_size=1, pin_memory=True, num_workers=4)

val_dataloader = None
train_meta(model,train_dataloader,val_dataloader,opt.epochs,opt.lr,opt.steps_til_summary,opt.epochs_til_checkpoint,model_dir,outer_loss, summary_fn,add_noise=opt.add_noise,clip=False)