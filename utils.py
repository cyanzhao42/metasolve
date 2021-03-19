import numpy as np
import torch 
import scipy
import os
import skimage.measure
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torchvision.utils import make_grid, save_image
from scipy.linalg import dft
import scipy.sparse
import scipy.signal
import imageio
import matplotlib.pyplot as plt

# --------------------------------------------
# credit /reference to https://github.com/vsitzmann/siren
# --------------------------------------------
def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dict_to_cuda(dict_in):
    out = {}
    for key, value in dict_in.items():
        if isinstance(value, torch.Tensor):
            out.update({key: value.cuda()})
        else:
            out.update({key: value})
    return out

def get_psnr(pred_img, gt_img):
    batch_size = pred_img.shape[0]

    pred_img = pred_img
    gt_img = gt_img

    p = pred_img
    trgt = gt_img
    ssim = skimage.measure.compare_ssim(
            p, trgt, multichannel=True, data_range=1)
    psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

    return ({"psnr":psnr},{"ssim":ssim})

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

def summary_fn(gt, observation, model_outputs, writer,total_steps,prefix="train",image_resolution=[64,64]):
    gt_img = []
    observation_img = []
    model_outputs_img = []
    init_vs_updates = []
    for i in range(len(gt)):
        gt_img.append(lin2img(gt[i], image_resolution))
        observation_img.append(lin2img(observation[i],image_resolution))
        model_outputs_img.append(lin2img(model_outputs[i],image_resolution))
        init_vs_updates.append(torch.cat(
                (gt_img[i].cuda(),observation_img[i],model_outputs_img[i]), dim=-1))

    init_vs_updates_allbatch = init_vs_updates[0]   
    for i in range(len(gt)-1):
        init_vs_updates_allbatch = torch.cat((init_vs_updates_allbatch,init_vs_updates[i+1]),dim=2)

    writer.add_image(prefix + 'init_vs_updates_allbatch', make_grid(init_vs_updates_allbatch, nrow=5, scale_each=False, normalize=True), global_step=total_steps)
    write_psnr(model_outputs_img,
               gt_img, writer, total_steps, prefix+'img_')

def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = len(pred_img)

    # pred_img = pred_img
    # gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i][0,:,:,:].detach().cpu().numpy()
        trgt = gt_img[i][0,:,:,:].detach().cpu().numpy()
        p = p.transpose(1, 2, 0)
        trgt = trgt.transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.measure.compare_ssim(
            p, trgt, multichannel=True, data_range=1)
        psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)
                   
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()