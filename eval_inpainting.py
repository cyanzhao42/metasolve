from torchvision.utils import make_grid, save_image
import torch
from torch.utils.data import DataLoader
from torchmeta.modules.utils import get_subdict_withkey
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
import skimage
import configargparse
import matplotlib.pyplot as plt
from utils import *
from meta_modules import *
from inner_modules import *
from train import *
import lpips
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False,
      is_config_file=True, help='Path to config file.')

p.add_argument('--note',type=str,default='omage90_g0.9_l1')
p.add_argument('--task',type=str,default='deblur')
p.add_argument('--gradient_type',type=str,default="static")
p.add_argument('--num_hidden_layers',type=int,default=3)
p.add_argument('--hidden_features',type=int,default=512)
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
p.add_argument('--epochs',type=int,default=9)
p.add_argument('--steps_til_summary',type=int,default=100)
p.add_argument('--epochs_til_checkpoint',type=int,default=10)
p.add_argument('--logging_root', type=str,
               default='/media/data2/cyanzhao/meta_solve/inpainting/ablation/', help='root for logging')
p.add_argument('--cuda',type=int,default=4)
opt = p.parse_args()

model_dir = "{}resol{}_face{}_csolver{}_ks{}_sig{}_noise{}_lr{}_innerlr{}_innerlrtyp{}_nummeta{}_numsolver{}_solverlr{}_data{}_sample{}_batch{}_note{}_g{}_hf512/".format(opt.logging_root,opt.resolution,opt.face,opt.class_solver,opt.kernel_size,opt.sigma,opt.add_noise,opt.lr,opt.inner_lr_init,opt.lr_type,opt.num_meta_steps,opt.num_updates,opt.gradient_steps,opt.datasetSize,opt.subsample,opt.batch_size,opt.note,opt.gradient_type)
print(model_dir)
shutil.copy('eval_inpainting_siren.py', model_dir + 'codes/eval_inpainting_siren.py')


torch.cuda.set_device(opt.cuda)
torch.manual_seed(0)

siren = SingleBVPNet(in_features=2, num_hidden_layers=opt.num_hidden_layers,hidden_features=opt.hidden_features, out_features=3).cuda()
image_resolution = [opt.resolution,opt.resolution]
A = inpainting_A_siren(image_resolution)
inner_loss_fn = partial(residual_inpainting_siren,A=A)
classical_solver = partial(gradient_decent,A=A)
# classical_solver = partial(cg_update,A=A)

model = meta_solver(siren, A, inner_loss_fn,opt.inner_lr_init,classical_solver,opt.num_updates,opt.gradient_steps,num_meta_steps=opt.num_meta_steps,lr_type=opt.lr_type,task=opt.task,gradient_type=opt.gradient_type,resolution=opt.resolution,final_gradient_steps=1,mode="eval")
model.cuda()
outer_loss_fn = partial(lploss,p=2)
# outer_loss_fn = partial(residual,A=A)
summary_fn = partial(summary_fn,image_resolution=image_resolution)

img_dataset_val = CelebA(
    split='val', datasetSize=1000, resolution=image_resolution,subsample_method=opt.subsample, downsampled=True,face=opt.face)
coord_dataset_val = Implicit2DWrapper(
    img_dataset_val, sidelength=image_resolution)
generalization_dataset_val = ImageGeneralizationWrapper(coord_dataset_val,test_sparsity='empty',train_sparsity_range=(90,100),generalization_mode='conv_cnp')
dataset_batch_val = task_batch(generalization_dataset_val,num=1)
val_dataloader = DataLoader(dataset_batch_val, shuffle=False,
                        batch_size=1, pin_memory=True, num_workers=0)


L1=[]
PSNRs=[]
PSNRs_blur=[]
SSIMs = []
lpips_loss = []
output_dir = os.path.join(model_dir, 'results')
utils.cond_mkdir(output_dir)

checkpoints_dir = os.path.join(model_dir, 'checkpoints')
PATH = "{}/model_current.pth".format(checkpoints_dir)
checkpoint = torch.load(PATH,map_location='cpu')
checkpoint = checkpoint['model_state_dict']
model.load_state_dict(checkpoint)
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
for val_idx, val_task_batch in enumerate(val_dataloader):
    model.load_state_dict((checkpoint))
    model.eval()
    model_outputs, gts, observations, inner_losses,progress,_= model(val_task_batch,opt.add_noise)
    outer_loss = outer_loss_fn(model_outputs=model_outputs,gts=gts,observation=observations)

    L1.append(outer_loss['loss'].detach().item())
    
    d = loss_fn_alex( utils.lin2img(model_outputs[0], image_resolution), utils.lin2img(gts[0], image_resolution))
    lpips_loss.append(d.detach().item())

    out_img = utils.lin2img(model_outputs[0], image_resolution).squeeze().permute(1, 2, 0).detach().cpu().numpy()
    out_img += 1
    out_img /= 2.
    out_img = np.clip(out_img, 0., 1.)

    gt_img = utils.lin2img(gts[0], image_resolution).squeeze().permute(1, 2, 0).detach().cpu().numpy()
    gt_img += 1
    gt_img /= 2.
    gt_img = np.clip(gt_img, 0., 1.)

    input_imag = utils.lin2img(observations[0], image_resolution).squeeze().permute(1, 2, 0).detach().cpu().numpy()
    input_imag += 1
    input_imag /= 2.
    input_imag = np.clip(input_imag, 0., 1.)
   
    if val_idx<32:
        imageio.imwrite(os.path.join(output_dir, 'painted{}.png'.format(val_idx)), out_img)
        imageio.imwrite(os.path.join(output_dir, 'gt{}.png'.format(val_idx)), gt_img)
        imageio.imwrite(os.path.join(output_dir, 'inpainting{}.png'.format(val_idx)), input_imag)
        imageio.imwrite(os.path.join(output_dir, 'sum{}.png'.format(val_idx)), np.concatenate([input_imag,out_img,gt_img]))
    PSNR = skimage.measure.compare_psnr(out_img, gt_img, data_range=1)
    SSIM = skimage.metrics.structural_similarity(out_img, gt_img, multichannel=True,data_range=1)
    PSNR_blur = skimage.measure.compare_psnr(input_imag, gt_img, data_range=1)
    PSNRs.append(PSNR)
    SSIMs.append(SSIM)
    PSNRs_blur.append(PSNR_blur)

PSNRs.append(sum(PSNRs)/len(PSNRs))
PSNRs.append(sum(PSNRs_blur)/len(PSNRs_blur))
SSIMs.append(sum(SSIMs)/len(SSIMs))
lpips_loss.append(sum(lpips_loss)/len(lpips_loss))
np.save(os.path.join(output_dir, 'PSNR' + '_context.npy'), PSNRs)  
np.save(os.path.join(output_dir, 'LPIPS' + '_context.npy'), lpips_loss)  

np.save(os.path.join(output_dir, 'l2' + '_context.npy'), L1)  
np.save(os.path.join(output_dir, 'SSIMs' + '_context.npy'), SSIMs)  
print(sum(L1)/len(L1))
print(sum(PSNRs)/len(PSNRs))
print(sum(PSNRs_blur)/len(PSNRs_blur))
print(sum(SSIMs)/len(SSIMs))
print(sum(lpips_loss)/len(lpips_loss))
