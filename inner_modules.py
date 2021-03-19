import matplotlib.pyplot as plt
import numpy as np
import torch
import dataio
import glob
import os
from torch.utils.data import DataLoader
import statistics
from tqdm.autonotebook import tqdm
import shutil
import imageio
from torchvision.utils import make_grid
import modules
import time


# --------------------------------------------
# Gaussian kernel for deblurring
# --------------------------------------------
def deblur_kernel(sigma,kernel_size):
    l=kernel_size
    sig=sigma
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel =  kernel / np.sum(kernel)
    return kernel

# --------------------------------------------
# Mask A for inpainting
# --------------------------------------------
def inpainting_A_siren(image_resolution,mode="onethird"):
    A = torch.ones(image_resolution)
    if mode=="onethird":
        A[int(image_resolution[0]/3)+1:int(image_resolution[0]/3*2)-1,int(image_resolution[0]/3)+1:int(image_resolution[0]/3*2-1)] = 0
    elif mode == "half": 
        A[int(image_resolution[0]/2)+1:,:] = 0
    elif mode == "half_left": 
        A[:,int(image_resolution[0]/2)+1:] = 0
    elif mode == "zero": 
        A[:,:] = 0
    else:
        raise NotImplementedError

    A = A.cuda()
    return A

def addnoise(inputs,add_noise):
     noise = torch.randn_like(inputs)*add_noise
     return inputs + noise

def pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :param dim: the dimension over which the tensors are padded
    :return:
    """

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")

        idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)

        idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x[idx], x], dim=d)
        pass

    return x

# --------------------------------------------
# residual for convolution with cyclic boundary condition
# --------------------------------------------
def residual_f(kernel,model_outputs,gts=None,observation=None,A=None):
    kernel = torch.tensor(kernel).cuda()
    pad=6
    if isinstance(model_outputs,list):
        channels = model_outputs[0].shape[2]
        counts = len(model_outputs)
        residual = torch.zeros(1).cuda()
        for j in range(counts):
            model_output_i = model_outputs[j].reshape(64,64,3)
            for i in range(channels):
                model_output = model_output_i[:,:,[i]]
                a = pad_circular_nd(model_output, pad=6,dim=[0,1])
                conved_output = torch.nn.functional.conv2d(input = a.unsqueeze(0).float().permute(0,3,1,2),weight=(kernel).unsqueeze(-1).unsqueeze(0).float().permute(0,3,1,2),stride=[1,1])
                conved_crop = conved_output[:,:,pad-2:76-pad-2,pad-2:76-pad-2].permute(0,2,3,1).reshape(1,-1,1)
                residual = residual + (((conved_crop-observation[j][:,:,[i]]))**2)
        residual = residual.mean()
        return {'loss':residual/channels/len(model_outputs)} 
    else:
        channels = model_outputs.shape[2]
        residual = torch.zeros(1).cuda()
        model_outputs = model_outputs.reshape(64,64,3)
        for i in range(channels):
            model_output = model_outputs[:,:,[i]]
            a = pad_circular_nd(model_output, pad=6,dim=[0,1])
            conved_output = torch.nn.functional.conv2d(input = a.unsqueeze(0).float().permute(0,3,1,2),weight=(kernel).unsqueeze(-1).unsqueeze(0).float().permute(0,3,1,2),stride=[1,1])
            conved_crop = conved_output[:,:,pad-2:76-pad-2,pad-2:76-pad-2].permute(0,2,3,1).reshape(1,-1,1)
            residual = residual + (((conved_crop-observation[:,:,[i]]))**2)
        residual = residual.mean()
        return {'loss':residual/channels} 

# --------------------------------------------
# residual give matrix A: \norm(Ax-y)
# --------------------------------------------
def residual(A,model_outputs,gts=None,observation=None):
    if isinstance(model_outputs,list):
        channels = model_outputs[0].shape[2]
        counts = len(model_outputs)
        residual = torch.zeros(1).cuda()
        for j in range(counts):
            for i in range(channels):
                residual = residual + (((A.matmul(model_outputs[j][:,:,[i]])-observation[j][:,:,[i]]))**2)
        residual = residual.mean()
        return {'loss':residual/channels/len(model_outputs)} 
    else:
        channels = model_outputs.shape[2]
        residual = torch.zeros(1).cuda()
        for i in range(channels):
            residual = residual + (((A.matmul(model_outputs[:,:,[i]])-observation[:,:,[i]]))**2)
        residual = residual.mean()
        return {'loss':residual/channels} 

# --------------------------------------------
# residual for inpainting task: \norm(Ax-y)
# --------------------------------------------
def residual_inpainting_siren(A,model_outputs,gts=None,observation=None):
    if isinstance(model_outputs,list):
        channels = model_outputs[0].shape[2]
        counts = len(model_outputs)
        residual = torch.zeros(1).cuda()
        for j in range(counts):
            for i in range(channels):
                residual = residual + ((A.view(1,-1,1)*((model_outputs[:,:,[i]])-observation[:,:,[i]]))**2)
        residual = residual.mean()
        return {'loss':residual/channels/len(model_outputs)} 
    else:
        channels = model_outputs.shape[2]
        residual = torch.zeros(1).cuda()
        for i in range(channels):
            residual = residual +((A.view(1,-1,1)*((model_outputs[:,:,[i]])-observation[:,:,[i]]))**2)
        residual = residual.mean()
        return {'loss':residual/channels} 

# --------------------------------------------
# lp loss
# --------------------------------------------
def lploss(model_outputs,gts=None,observation=None,p=1,A=None):
    if A==None:
        loss = 0
        for i in range(len(model_outputs)):
            loss = loss + ((torch.abs(model_outputs[i]-gts[i]))**p).mean()
        return {'loss':loss/len(model_outputs)}
    else:
        loss = 0
        for i in range(len(model_outputs)):
            loss = loss + ((A.view(1,-1,1)*(torch.abs(model_outputs[i]-gts[i])))**p).mean()
        return {'loss':loss/len(model_outputs)}

# --------------------------------------------
# lpips loss
# --------------------------------------------   
def lpips_loss(model_outputs,loss_fn=None,gts=None,observation=None,A=None):
    loss = 0
    for i in range(len(model_outputs)):
        model_output=model_outputs[i].reshape(64,64,3).unsqueeze(0).permute(0,3,1,2)
        gt=gts[i].reshape(64,64,3).unsqueeze(0).permute(0,3,1,2)
        loss += loss_fn.forward(model_output,gt)
    return {'loss':loss/len(model_outputs)}

# --------------------------------------------
# gradient descent 
# --------------------------------------------
def gradient_decent(A,model_out,observation,gradient_steps,num_updates,**kwargs):
    progress = []
    progress.append(model_out)
    #model_out: 1,resol*resol,channel
    #A: resol*resol, resol*resol
    #obervation: 1,resol*resol,channel
    if num_updates==0:
        return model_out,progress
    x_rgb = torch.zeros((model_out.shape)).cuda()
    # alpha = gradient_steps*torch.ones(num_updates)
    if isinstance(gradient_steps,int) or isinstance(gradient_steps,float):
        alpha = gradient_steps*torch.ones(num_updates).cuda()
    else:
        alpha = gradient_steps
    for channelnum in range(model_out.shape[2]):
        x = model_out[:,:,channelnum].unsqueeze(-1)
        b = observation[:,:,channelnum].unsqueeze(-1)
        for i in range(num_updates):
            gradient = A.view(1,-1,1)*(A .view(1,-1,1)* x - b)
            x = x-alpha[i]*gradient
        x_rgb[:,:,channelnum] = x.squeeze(-1)
    progress.append(x_rgb)
    return x_rgb,progress

# --------------------------------------------
# RED with fixed point method
# --------------------------------------------
def gradient_decent_ircnn_fp(A,model_out,observation,gradient_steps,num_updates,denoiser,fft_Ht_y_torch,fft_HtH_torch):
    progress = []
    progress.append(model_out)
    l=0.02
    #model_out: 1,resol*resol,channel
    #A: resol*resol, resol*resol
    #obervation: 1,resol*resol,channel
    if num_updates==0:
        return model_out,progress
    x_rgb = torch.zeros((model_out.shape)).cuda()
    # alpha = gradient_steps*torch.ones(num_updates)
    if isinstance(gradient_steps,int) or isinstance(gradient_steps,float):
        alpha = gradient_steps*torch.ones(num_updates).cuda()
    else:
        alpha = gradient_steps
    x0 = model_out.cuda()
    for i in range(num_updates):
        f_x_est = denoiser(x0.view(64,64,3).unsqueeze(0).permute(0,3,1,2)).permute(0,2,3,1).squeeze(0).reshape(-1,3)
        # progress.append(x_rgb)
        f_x_est = f_x_est.reshape(64,64,3)
        # fft_est = torch.zeros([64, 64, 3,2]).cuda()
        # fft_est[:,:,0,:] = torch.rfft(f_x_est[:,:,0],2,onesided=False)
        # fft_est[:,:,1,:] = torch.rfft(f_x_est[:,:,1],2,onesided=False)
        # fft_est[:,:,2,:] = torch.rfft(f_x_est[:,:,2],2,onesided=False)
        b_torch = fft_Ht_y_torch.cuda() + l*torch.cat([torch.rfft(f_x_est[:,:,0],2,onesided=False).unsqueeze(2),torch.rfft(f_x_est[:,:,1],2,onesided=False).unsqueeze(2),torch.rfft(f_x_est[:,:,2],2,onesided=False).unsqueeze(2)],dim=2)
        A_matrix_torch =(fft_HtH_torch + l).unsqueeze(-1)
        x_est_div = b_torch/A_matrix_torch
        x_rgb = torch.cat([torch.irfft(x_est_div[:,:,0,:], 2, onesided=False).view(1,-1,1),torch.irfft(x_est_div[:,:,1,:], 2, onesided=False).view(1,-1,1),torch.irfft(x_est_div[:,:,2,:], 2, onesided=False).view(1,-1,1)],dim=2)
        # x_rgb[:,:,0] = torch.irfft(x_est_div[:,:,0,:], 2, onesided=False).view(1,-1,1)
        # x_rgb[:,:,1] = torch.irfft(x_est_div[:,:,1,:], 2, onesided=False).view(1,-1,1)
        # x_rgb[:,:,2] = torch.irfft(x_est_div[:,:,2,:], 2, onesided=False).view(1,-1,1)
        x0 = x_rgb.float().reshape(1,-1,3)
        progress.append(x_rgb)
    return x0,progress


