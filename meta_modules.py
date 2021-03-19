import torch
from torch import nn
from collections import OrderedDict
from modules import *
import numpy as np
from utils import *
import time
import numpy.fft as fft
from denoiser.denoiser import DnCNN
from inner_modules import *
# --------------------------------------------
# code take reference from https://github.com/vsitzmann/metasdf
# --------------------------------------------

class meta_solver(nn.Module):
    def __init__(self,siren, A, inner_loss_fn,inner_lr_init,classical_solver,num_updates,gradient_steps,num_meta_steps=3,lr_type="none",task="deblur",gradient_type="static",grey=False,mode="train",oversample=4,firstorder=False,resolution=32,gradient_limit=5,randomA=False,final_gradient_steps=None,tunedenoiser=False,xyencoder=False,kernel=None,denoiser=None):
        super().__init__()
        self.siren = siren
        self.resolution = resolution
        
        self.randomA = randomA
    
        self.A = A
        self.inner_loss_fn = inner_loss_fn
        self.num_meta_steps = num_meta_steps
        self.classical_solver = classical_solver
        self.num_updates = num_updates
        self.gradient_steps = gradient_steps
        self.num_meta_steps = num_meta_steps
        self.task = task
        self.grey = grey
        self.lr_type = lr_type
        self.gradient_type = gradient_type
        self.mode = mode
        self.oversample = oversample
        self.firstorder=firstorder
        self.gradient_limit = gradient_limit
        self.denoiser_type = denoiser
        if final_gradient_steps==None:
            self.final_gradient_steps = gradient_steps
        else:
            self.final_gradient_steps = final_gradient_steps

        if self.lr_type == "static":
            self.register_buffer('lr',torch.Tensor([inner_lr_init]))
        elif self.lr_type == "global":
            self.lr=nn.Parameter(torch.Tensor([inner_lr_init]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([inner_lr_init]))
                                            for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter':
            self.lr = nn.ModuleList([])
            siren_params = siren.parameters()
            for param in siren_params:
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * inner_lr_init)
                                                    for _ in range(num_meta_steps)]))
        self.fft_HtH_torch  = None
        self.fft_psf = None

        if self.gradient_type == "per_step":
            self.gradient_steps = nn.Parameter(torch.ones(num_updates)*gradient_steps,requires_grad=True)
        else:
            self.gradient_steps = torch.ones(num_updates)*gradient_steps

        self.denoiser=None
        
        if task=="deblur_dncnn_fp":
            map_location = {'cuda:5':'cuda:{}'.format(torch.cuda.current_device())}
            self.denoiser = DnCNN().cuda()
            checkpoint = torch.load("model_zoo/denoiser/dncnn.ckpt",map_location)
            self.denoiser.load_state_dict(checkpoint["state_dict"])

            if not tunedenoiser:
                self.denoiser.eval()
                for params in self.denoiser.parameters():
                    params.requires_grad=False
            else:
                self.denoiser.apply(set_bn_eval)
 
            #precomputation for RED-FP
            fft_psf = np.zeros([self.resolution,self.resolution,3])
            t = np.floor(5/2)
            H=self.resolution
            W=self.resolution
            for i in range(3):
                fft_psf[int(H/2+1-t-1):int(H/2+1+t),int(W/2+1-t-1):int(W/2+1+t),i]=kernel
            fft_psf = fft.fft2(fft.fftshift(fft_psf,axes=(0,1)),axes=(0,1))
            self.fft_psf = (fft_psf)
            self.sigma = 7.65/2
            self.fft_HtH_torch = torch.tensor(np.abs(fft_psf)**2/self.sigma**2).cuda()
           
        self.xyencoder = xyencoder
        if self.xyencoder:
            self.position_encoder = nn.Parameter(torch.rand(1,resolution*resolution,32),requires_grad=True)
        self.kernel = kernel

    #helper function for convolution with cyclic boundary condition
    def A_gauss(self,img,kernel):
        img = img.squeeze(0).reshape(self.resolution,self.resolution,3)
        [H,W,channel]=img.shape
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        image2 = np.zeros([H,W,channel])
        for c in range(channel):
            image2[:,:,c]= scipy.signal.convolve2d(img[:,:,c],kernel,mode='same',boundary='wrap')
        return torch.tensor(image2).float().unsqueeze(0).view(1,-1,3).cuda()

    def forward(self,task_batch,add_noise=0,**kwargs):
        model_outputs=[]
        gts = []
        observations = []
        inner_precedures = []
        inner_losses = torch.zeros(self.num_meta_steps)
        for task_idx in range(len(task_batch[1])):
            gt = task_batch[1][task_idx]
            model_input = task_batch[0]
            model_input = dict_to_cuda(model_input)
            gt = dict_to_cuda(gt)
            fft_Ht_y_torch = None
            if self.task=="deblur" or self.task=='inpainting' or self.task=="deblur_dncnn_fp":
                if self.randomA:
                    
                    #uncommand during training:
                    # A = torch.rand(self.resolution,self.resolution).cuda()
                    # self.A = (A>((torch.rand(1).cuda()+0.1)/1.2))*1
                    # self.A = (A>((torch.rand(1).cuda())))*1
                    
                    #uncommand during testing:
                    a = torch.randperm(32*32)
                    self.A = torch.zeros(self.resolution*self.resolution).cuda()
                    self.A[a[0:500]]=1 #change 500 to other number if needed
                    self.A = self.A.reshape(32,32)
                    
                if self.A.shape[0]==self.resolution:
                    observation = gt['img']*0
                    observation[0,:,0] = self.A.view(-1)*(gt['img'][0,:,0])
                    observation[0,:,1] = self.A.view(-1)*(gt['img'][0,:,1])
                    observation[0,:,2] = self.A.view(-1)*(gt['img'][0,:,2])
                    observation = addnoise(observation,add_noise)
                    observations.append(observation)
                    gts.append(gt['img'])
                    
                elif self.denoiser_type=="dncnn":
                    observation = self.A_gauss(img=gt['img'],kernel = self.kernel)
                    observation = addnoise(observation,add_noise)
                    observations.append(observation)
                    gts.append(gt['img'])
                    
                    fft_y = fft.fft2(observation.view(self.resolution,self.resolution,3).cpu().numpy(),axes=(0,1))
                    fft_Ht_y = np.conjugate(self.fft_psf)*fft_y/self.sigma**2
                    fft_Ht_y_torch = torch.zeros([self.resolution, self.resolution, 3,2])
                    fft_Ht_y_torch[:,:,:,0]=torch.tensor(np.real(fft_Ht_y))
                    fft_Ht_y_torch[:,:,:,1]=torch.tensor(np.imag(fft_Ht_y))
               
            if self.xyencoder:
                model_input['coords'] = torch.cat([model_input['coords'],self.position_encoder],2)
            updated_aparams, inner_loss,inner_precedure1= self.generate_params(gt,observation,model_input,fft_Ht_y_torch=fft_Ht_y_torch,fft_HtH_torch=self.fft_HtH_torch)
            model_output,inner_precedure2= self.forward_with_params(observation=observation,model_input=model_input, updated_aparams=updated_aparams,fft_Ht_y_torch=fft_Ht_y_torch,fft_HtH_torch=self.fft_HtH_torch)
            inner_precedures.append(inner_precedure1)
            inner_precedures.append(inner_precedure2)
            model_outputs.append(model_output)
            inner_losses = inner_loss + inner_losses
        return model_outputs, gts, observations, inner_losses/len(task_batch[1]),inner_precedures,updated_aparams

    def generate_params(self,gt,observation,model_input,fft_Ht_y_torch=None,fft_HtH_torch=None):
        inner_losses = torch.zeros(self.num_meta_steps)
        if self.siren!=None:
            updated_params = OrderedDict((name, param) for (
                            name, param) in self.siren.named_parameters())
        else:
            updated_params = None

        inner_precedures = []
        for i in range(self.num_meta_steps):
            self.siren.zero_grad()
            model_output = self.siren(model_input,params=updated_params)
            if self.num_updates>0:
                solver_output,progress= self.classical_solver(model_out=model_output['model_out'],observation=observation,gradient_steps=self.gradient_steps,num_updates=self.num_updates,denoiser = self.denoiser,fft_Ht_y_torch=fft_Ht_y_torch,fft_HtH_torch=fft_HtH_torch)
            else:
                solver_output = model_output['model_out']
                progress = model_output['model_out'].detach()
            inner_precedures.append(progress)
            inner_loss =  self.inner_loss_fn(A=self.A,model_outputs=solver_output,observation=observation,gts=gt)
            inner_losses[i]=inner_loss['loss']
            
            inner_loss = inner_loss['loss']

            if self.mode=="eval" or self.firstorder:
                grads = torch.autograd.grad(inner_loss, updated_params.values(),
                                        create_graph=False,allow_unused=True)
            else:
                grads = torch.autograd.grad(inner_loss, updated_params.values(),
                                        create_graph=True,allow_unused=True)

            for j, ((name, param), grad) in enumerate(zip(updated_params.items(), grads)): 
                if self.lr_type in ['static', 'global']:
                    lr = self.lr
                elif self.lr_type in ['per_step']:
                    lr = self.lr[i]
                elif self.lr_type in ['per_parameter']:
                    lr = self.lr[j][i] if i < self.num_meta_steps else 1e-2
                else:
                    raise NotImplementedError
                
                if not(self.gradient_limit==None):
                    if torch.norm(grad)>self.gradient_limit:
                        grad = grad/torch.norm(grad)*self.gradient_limit
                #add to avoid gradient explosion 
                # if torch.norm(grad)>0.1:
                #         grad = grad/torch.norm(grad)*0.01
                #         print("clip_gradient")
                updated_params[name] = param - lr * grad
        return updated_params, inner_losses,inner_precedures

    
    def forward_with_params(self,observation,model_input, updated_aparams,fft_Ht_y_torch=None,fft_HtH_torch=None):
        if self.siren==None:
            model_output = {'model_out':observation}
        else:
            model_output = self.siren(model_input,params = updated_aparams)
            
        if self.num_updates>0:
            solver_output,progress= self.classical_solver(model_out=model_output['model_out'],observation=observation,gradient_steps=self.final_gradient_steps,num_updates=self.num_updates,denoiser = self.denoiser,fft_Ht_y_torch=fft_Ht_y_torch,fft_HtH_torch=fft_HtH_torch)
        else:
            solver_output = model_output['model_out']
            progress = None
            
        return solver_output,progress


        