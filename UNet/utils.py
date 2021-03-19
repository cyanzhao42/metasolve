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

def deblur_A(image_resolution,sigma,kernel_size,bdd_con="fill0"):
    l=kernel_size
    sig=sigma
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel =  kernel / np.sum(kernel)
    resol = image_resolution[0]
    print(kernel)
    A = scipy.sparse.eye(resol**2)*0
    centerdiag = np.ones([resol**2])
    diagvec=list()
    kval=list()
    for i in range(l):
        for j in range(l):
            diagvec.append(kernel[i,j])
            kval.append((i-int(l/2))*resol+(j-int(l/2)))
    A = scipy.sparse.diags(diagvec,kval,[resol**2,resol**2])
    A = (A.todense())
    if bdd_con == "fill0":
        for row in range(resol):
            for i in range(int((l-1)/2)):
                bdd_row=(row*resol+i)
                for k in range(int((l-1)/2)-i):
                    for p in range(l):
                        if bdd_row+resol*(p-int((l-1)/2))-k-i-1>=0 and bdd_row+resol*(p-int((l-1)/2))-k-1-i<resol**2:
                            A[bdd_row,bdd_row+resol*(p-int((l-1)/2))-k-i-1]=0
                bdd_row=((row+1)*resol-i-1)
                for k in range(int((l-1)/2)-i):
                    for p in range(l):
                        if bdd_row+resol*(p-int((l-1)/2))+k+i+1>=0 and bdd_row+resol*(p-int((l-1)/2))+k+i+1<resol**2:
                            A[bdd_row,bdd_row+resol*(p-int((l-1)/2))+i+k+1]=0
    else:
        raise NotImplementedError
    #### check #####
    x = np.ones([resol**2])
    y1 = A @ x
    y2 = scipy.signal.convolve2d(x.reshape([resol,resol]),kernel,mode="same",boundary="fill")
    if ((y1.reshape([resol,resol])-y2).max())>1e-10:
        print("A is wrong")
        raise 
    A = torch.from_numpy(A).float().cuda()
    return A

def inpainting_A(image_resolution,mode="onethird"):
    A = torch.ones(image_resolution)
    if mode=="onethird":
        A[int(image_resolution[0]/3)+1:int(image_resolution[0]/3*2)-1,int(image_resolution[0]/3)+1:int(image_resolution[0]/3*2-1)] = 0
    else: 
        raise NotImplementedError

    A = torch.diag(A.view(-1),0).cuda()
    return A



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
    elif mode>0:
        a = torch.randperm(32*32)
        A = torch.zeros(image_resolution[0]*image_resolution[1]).cuda()
        A[a[0:500]]=1
        A = A.reshape(32,32)
    else:
        raise NotImplementedError

    A = A.cuda()
    return A


def phase_A(image_resolution):
    dftmtx = np.fft.fft(np.eye(image_resolution[0]))
    A = np.kron(dftmtx.T,dftmtx)
    A_real = torch.tensor(A.real).float().cuda()
    A_imag = torch.tensor(A.imag).float().cuda()
    return [A_real, A_imag]


def addnoise(inputs,add_noise):
     noise = torch.randn_like(inputs)*add_noise
     return inputs + noise


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

def lin2img_pr(tensor, image_resolution=None):
    resolution_h,resolution_w, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        h = image_resolution[0]
        w = image_resolution[1]
    image = tensor[int(resolution_h/2-(h/2-1)):int(resolution_h/2+1+(h/2)),int(resolution_w/2-(w/2-1)):int(resolution_w/2+1+(w/2))]
    return image.unsqueeze(0).permute(0,3,1,2)
    # return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


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

def summary_fn_pr(gt, observation, model_outputs, writer,total_steps,prefix="train",image_resolution=[64,64]):
    gt_img = []
    observation_img = []
    model_outputs_img = []
    init_vs_updates = []
    diff = []
    for i in range(len(gt)):
        gt_img.append(lin2img_pr(gt[i], image_resolution))
        observation_img.append(lin2img_pr(observation[i],image_resolution))
        model_outputs_img.append(lin2img_pr(model_outputs[i],image_resolution))
        init_vs_updates.append(torch.cat(
                (gt_img[i].cuda(),observation_img[i],model_outputs_img[i]), dim=-1))
        diff.append(torch.cat(
                (gt_img[i].cuda(),model_outputs_img[i]), dim=-1))

    init_vs_updates_allbatch = init_vs_updates[0]   
    diff_all = diff[0] 
    for i in range(len(gt)-1):
        init_vs_updates_allbatch = torch.cat((init_vs_updates_allbatch,init_vs_updates[i+1]),dim=2)
        diff_all = torch.cat((diff_all,diff[i+1]),dim=2)

    writer.add_image(prefix + 'init_vs_updates_allbatch', make_grid(init_vs_updates_allbatch, nrow=5, scale_each=False, normalize=True), global_step=total_steps)
    writer.add_image(prefix + 'diff', make_grid(diff_all, nrow=5, scale_each=False, normalize=True), global_step=total_steps)
    write_psnr(model_outputs_img,
               gt_img, writer, total_steps, prefix+'img_')


def summary_fn_pr_cdp(gt, observation, model_outputs, writer,total_steps,prefix="train",image_resolution=[64,64]):
    gt_img = []
    observation_img = []
    model_outputs_img = []
    init_vs_updates = []
    diff = []
    for i in range(len(gt)):
        gt_img.append(lin2img_pr(gt[i].squeeze(0).reshape(32,32,1), image_resolution))
        # observation_img.append(lin2img_pr(observation[i].squeeze(0).reshape(32,32,1), image_resolution))
        model_outputs_img.append(lin2img_pr(model_outputs[i].squeeze(0).reshape(32,32,1), image_resolution))
        diff.append(torch.cat(
                (gt_img[i].cuda(),model_outputs_img[i]), dim=-1))

    # init_vs_updates_allbatch = init_vs_updates[0]   
    diff_all = diff[0] 
    for i in range(len(gt)-1):
        # init_vs_updates_allbatch = torch.cat((init_vs_updates_allbatch,init_vs_updates[i+1]),dim=2)
        diff_all = torch.cat((diff_all,diff[i+1]),dim=2)

    # writer.add_image(prefix + 'init_vs_updates_allbatch', make_grid(init_vs_updates_allbatch, nrow=5, scale_each=False, normalize=True), global_step=total_steps)
    writer.add_image(prefix + 'diff', make_grid(diff_all, nrow=5, scale_each=False, normalize=True), global_step=total_steps)
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





def summary_progress(progress,gt_img,output_dir,val_idx,image_resolution):
    totalimages = []
    fig = plt.figure()
    for p in range(1):
        for i in range(5):
            images = []
            for j in range(6):
                imag = progress[p][i][j]
                imag = lin2img_pr(imag, image_resolution).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                imag += 1
                imag /= 2.
                imag = np.clip(imag, 0., 1.)
                plt.subplot(6,6,i*6+j+1)
                plt.imshow(imag.squeeze(2))
                images.append(imag)
            for k in range(6):
                if k==0:
                    permetastep = images[0]
                else:
                    permetastep = np.concatenate([permetastep,images[k]],axis=0)
            totalimages.append(permetastep)
        for p in range(5):
            if p==0:
                image = totalimages[0]
            else:
                image = np.concatenate([image,totalimages[p]],axis=1)
        plt.subplot(7,7,49)
        plt.imshow(gt_img.squeeze(2))
    # imageio.imwrite(os.path.join(output_dir, 'progress{}.png'.format(val_idx)),figure1)
    # plt.imshow(image.squeeze(-1))
    plt.savefig(os.path.join(output_dir, 'progress{}.png'.format(val_idx)))

            



def summary_fn_unet(gt, observation, model_outputs, writer,total_steps,prefix="train",image_resolution=[64,64]):
    gt_img = []
    observation_img = []
    model_outputs_img = []
    init_vs_updates = []
    gt_img.append(lin2img(gt, image_resolution))
    observation_img.append(lin2img(observation,image_resolution))
    model_outputs_img.append(lin2img(model_outputs,image_resolution))
    init_vs_updates.append(torch.cat(
                (gt_img[0].cuda(),observation_img[0],model_outputs_img[0]), dim=-1))


    writer.add_image(prefix + 'init_vs_updates_allbatch', make_grid(init_vs_updates[0], nrow=5, scale_each=False, normalize=True), global_step=total_steps)
    write_psnr(model_outputs_img,
               gt_img, writer, total_steps, prefix+'img_')
