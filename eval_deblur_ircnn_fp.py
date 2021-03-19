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

p.add_argument('--note',type=str,default='90_lpips_vgg_4')
p.add_argument('--denoiser',type=str,default='dncnn')
p.add_argument('--tunedenoiser',type=bool,default=True)
p.add_argument('--xyencoder',type=bool,default=False)
p.add_argument('--gradient_type',type=str,default="static")
p.add_argument('--task',type=str,default='deblur_dncnn_fp')
p.add_argument('--num_hidden_layers',type=int,default=3)
p.add_argument('--hidden_features',type=int,default=256)
p.add_argument('--resolution',type=int,default=64)
p.add_argument('--kernel_size',type=int,default=5)
p.add_argument('--sigma',type=float,default=1.5)
p.add_argument('--add_noise',type=float,default=7.65/255.)
p.add_argument('--lr',type=float,default=1e-5)
p.add_argument('--inner_lr_init',type=float,default=1)
p.add_argument('--lr_type',type=str,default="per_parameter")
p.add_argument('--num_meta_steps',type=int,default=3)
p.add_argument('--class_solver',type=str,default="gd")
p.add_argument('--gradient_steps',type=float,default=1)
p.add_argument('--num_updates',type=int,default=3)
p.add_argument('--datasetSize',type=int,default=150000)
p.add_argument('--subsample',type=str,default='smooth')
p.add_argument('--face',type=bool,default=True)
p.add_argument('--batch_size',type=int,default=3)
p.add_argument('--epochs',type=int,default=3)
p.add_argument('--steps_til_summary',type=int,default=5)
p.add_argument('--epochs_til_checkpoint',type=int,default=1)
p.add_argument('--logging_root', type=str,
               default='/media/data2/cyanzhao/meta_solve/debluring_ircnn_fp/ablation2_sigma7.65div2/', help='root for logging')
p.add_argument('--cuda',type=int,default=7)
opt = p.parse_args()

# model_dir = "{}resol{}_face{}_csolver{}_ks{}_sig{}_noise{}_lr{}_innerlr{}_innerlrtyp{}_nummeta{}_numsolver{}_solverlr{}_data{}_sample{}_batch{}_tunedenoiser{}_xyencoder{}/".format(opt.logging_root,opt.resolution,opt.face,opt.class_solver,opt.kernel_size,opt.sigma,opt.add_noise,opt.lr,opt.inner_lr_init,opt.lr_type,opt.num_meta_steps,opt.num_updates,opt.gradient_steps,opt.datasetSize,opt.subsample,opt.batch_size,opt.tunedenoiser,opt.xyencoder)
# model_dir = "{}resol{}_face{}_csolver{}_ks{}_sig{}_noise{}_lr{}_innerlr{}_innerlrtyp{}_nummeta{}_numsolver{}_solverlr{}_data{}_sample{}_batch{}/".format(opt.logging_root,opt.resolution,opt.face,opt.class_solver,opt.kernel_size,opt.sigma,opt.add_noise,opt.lr,opt.inner_lr_init,opt.lr_type,opt.num_meta_steps,opt.num_updates,opt.gradient_steps,opt.datasetSize,opt.subsample,opt.batch_size)
# model_dir = "{}resol{}_face{}_csolver{}_ks{}_sig{}_noise{}_lr{}_innerlr{}_innerlrtyp{}_nummeta{}_numsolver{}_solverlr{}_data{}_sample{}_batch{}_tunedenoiser{}_xyencoder{}_denoiser{}_note{}/".format(opt.logging_root,opt.resolution,opt.face,opt.class_solver,opt.kernel_size,opt.sigma,opt.add_noise,opt.lr,opt.inner_lr_init,opt.lr_type,opt.num_meta_steps,opt.num_updates,opt.gradient_steps,opt.datasetSize,opt.subsample,opt.batch_size,opt.tunedenoiser,opt.xyencoder,opt.denoiser,opt.note)
model_dir = "{}resol{}_face{}_csolver{}_ks{}_sig{}_noise{}_lr{}_innerlr{}_innerlrtyp{}_nummeta{}_numsolver{}_solverlr{}_data{}_sample{}_batch{}_tunedenoiser{}_xyencoder{}_denoiser{}_note{}/".format(opt.logging_root,opt.resolution,opt.face,opt.class_solver,opt.kernel_size,opt.sigma,opt.add_noise,opt.lr,opt.inner_lr_init,opt.lr_type,opt.num_meta_steps,opt.num_updates,opt.gradient_steps,opt.datasetSize,opt.subsample,opt.batch_size,opt.tunedenoiser,opt.xyencoder,opt.denoiser,opt.note)

print(model_dir)
shutil.copy('eval_deblur_ircnn_fp.py', model_dir + 'codes/eval_deblur_ircnn_fp.py')


torch.cuda.set_device(opt.cuda)
torch.manual_seed(0)

if opt.xyencoder:
    siren = SingleBVPNet(in_features=2+32, num_hidden_layers=opt.num_hidden_layers,hidden_features=opt.hidden_features, out_features=3).cuda()
else:
    siren = SingleBVPNet(in_features=2, num_hidden_layers=opt.num_hidden_layers,hidden_features=opt.hidden_features, out_features=3).cuda()
image_resolution = [opt.resolution,opt.resolution]
# A = deblur_A(image_resolution,opt.sigma,opt.kernel_size)
A,kernel = deblur_A(image_resolution,opt.sigma,opt.kernel_size)

inner_loss_fn = partial(residual_f,kernel=kernel)
classical_solver = partial(gradient_decent_ircnn_fp,A=A)
# classical_solver = partial(cg_update,A=A)

model = meta_solver(siren, A, inner_loss_fn,opt.inner_lr_init,classical_solver,opt.num_updates,opt.gradient_steps,num_meta_steps=opt.num_meta_steps,lr_type=opt.lr_type,task=opt.task,gradient_limit=None,firstorder=False,gradient_type=opt.gradient_type,tunedenoiser=opt.tunedenoiser,xyencoder=opt.xyencoder,resolution=opt.resolution,mode="eval",denoiser="dncnn",kernel=kernel)
model.cuda()
outer_loss_fn = partial(lploss,p=2)
# outer_loss_fn = partial(residual,A=A)
summary_fn = partial(summary_fn,image_resolution=image_resolution)

img_dataset_val = CelebA(
    split='val', datasetSize=500, resolution=image_resolution,subsample_method=opt.subsample, downsampled=True,face=opt.face)
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
lpips_loss_vgg = []
lpips_loss_alex = []
output_dir = os.path.join(model_dir, 'results')
utils.cond_mkdir(output_dir)

checkpoints_dir = os.path.join(model_dir, 'checkpoints')
PATH = "{}/model_current.pth".format(checkpoints_dir)
checkpoint = torch.load(PATH,map_location='cpu')
checkpoint = checkpoint['model_state_dict']
model.load_state_dict(checkpoint)
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

for val_idx, val_task_batch in enumerate(val_dataloader):
    model.load_state_dict((checkpoint))
    model.eval()
    model_outputs, gts, observations, inner_losses,_,_= model(val_task_batch,opt.add_noise)
    # outer_loss = outer_loss_fn(model_outputs=model_outputs,gts=gts,observation=observations)

    # L1.append(outer_loss['loss'].detach().item())
    
    d = loss_fn_alex( utils.lin2img(model_outputs[0], image_resolution), utils.lin2img(gts[0], image_resolution))
    lpips_loss_alex.append(d.detach().item())

    d = loss_fn_vgg( utils.lin2img(model_outputs[0], image_resolution), utils.lin2img(gts[0], image_resolution))
    lpips_loss_vgg.append(d.detach().item())

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
   
    L1.append(((gt_img-out_img)**2).mean())
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
lpips_loss_vgg.append(sum(lpips_loss_vgg)/len(lpips_loss_vgg))
lpips_loss_alex.append(sum(lpips_loss_alex)/len(lpips_loss_alex))

np.save(os.path.join(output_dir, 'PSNR' + '_context.npy'), PSNRs)  
np.save(os.path.join(output_dir, 'LPIPS_alex' + '_context.npy'), lpips_loss_alex)  
np.save(os.path.join(output_dir, 'LPIPS_vgg' + '_context.npy'), lpips_loss_vgg)  

np.save(os.path.join(output_dir, 'l2' + '_context.npy'), L1)  
np.save(os.path.join(output_dir, 'SSIMs' + '_context.npy'), SSIMs)  
print(sum(L1)/len(L1))
print(sum(PSNRs)/len(PSNRs))
print(sum(PSNRs_blur)/len(PSNRs_blur))
print(sum(SSIMs)/len(SSIMs))
print(sum(lpips_loss_vgg)/len(lpips_loss_vgg))
print(sum(lpips_loss_alex)/len(lpips_loss_alex))

# for val_idx, val_task_batch in enumerate(val_dataloader):
#     clean_img = val_task_batch[1][0]['img']
#     noise = torch.randn(clean_img.size()).mul_(opt.add_noise).cuda()
#     y = clean_img+(noise)
#     denoised = model.denoiser(y.view(64,64,3).unsqueeze(0).permute(0,3,1,2))

# f = plt.figure(figsize=[50,50])
# axes = []
# for i in range(1):
#     for j in range(3):
#         for k in range(4):
#             image = progress[i][j][k]
#             out_img = utils.lin2img(image.reshape([1,4096,3]), image_resolution).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#             out_img += 1
#             out_img /= 2.
#             out_img = np.clip(out_img, 0., 1.)
#             axes.append(f.add_subplot(4,4, j*4+k+1))
#             plt.imshow(out_img)
# for k in range(4):
#             image = progress[1][k]
#             out_img = utils.lin2img(image.reshape([1,4096,3]), image_resolution).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#             out_img += 1
#             out_img /= 2.
#             out_img = np.clip(out_img, 0., 1.)
#             axes.append(f.add_subplot(4,4, 3*4+k+1))
#             plt.imshow(out_img)
# f.tight_layout() 
# plt.show()
# plt.savefig("test.png")

