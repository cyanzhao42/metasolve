import torch
from torch import nn
from torchmeta.modules import MetaModule
from torchmeta.modules import MetaSequential
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
from functools import partial
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)

# --------------------------------------------
# SIREN 
# --------------------------------------------
# credit to https://github.com/vsitzmann/siren
# --------------------------------------------
class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None,**kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(
            *[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

class Sine(nn.Module):
    def __init__(self,omega0=90):
        super().__init__()
        self.omega0=omega0

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 90
        return torch.sin(self.omega0 * input)

class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None,omega0=90):
        super().__init__()

        self.first_layer_init = None
        self.omega0 = omega0

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init
            
        self.weight_init = partial(self.weight_init,omega0=omega0)
        self.nl = partial(nl,omega0=self.omega0)
        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        # Apply special initialization to first layer, if applicable.
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

def sine_init(m,omega0=90):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 90
            m.weight.uniform_(-np.sqrt(6 / num_input) / omega0,
                              np.sqrt(6 / num_input) / omega0)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 90
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3,omega0=90,**kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type,omega0=omega0)
        self.omega0 = omega0
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        # coords_org = model_input['coords'].clone(
        # ).detach().requires_grad_(True)
        coords_org = model_input['coords']
        coords = coords_org

        output = self.net(coords, get_subdict(params, 'net'),omega0=self.omega0)
        return {'model_in': coords_org, 'model_out': output}


# --------------------------------------------
# FILMED SIREN
# --------------------------------------------
# Some extension - Filmed SIREN (not presented in the final results)
class filmed_siren(MetaModule):
    def __init__(self, resolution=[32,32],out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.siren = SingleBVPNetfilmed(in_features=2, num_hidden_layers=num_hidden_layers,hidden_features=hidden_features, out_features=out_features).cuda()
        self.encoder = CNN(resolution=resolution, out_features=256,channel_num=out_features)
        self.filmGenerator = LinearFilmGen(in_features=hidden_features, out_features=hidden_features * 2,bias=True)

    def forward(self, model_input,observation, params=None):
        embedding = self.encoder(observation,params=self.get_subdict(params, 'encoder'))
        gamma,beta = self.filmGenerator(embedding,params=self.get_subdict(params, 'filmGenerator'))
        output = self.siren(model_input,gamma,beta,params=self.get_subdict(params, 'siren'))
        return output

class filmed_siren_noencoder(MetaModule):
    def __init__(self, resolution=[32,32],out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.siren = SingleBVPNetfilmed(in_features=2, num_hidden_layers=num_hidden_layers,hidden_features=hidden_features, out_features=out_features).cuda()
  
    def forward(self, model_input,observation,gamma,beta, params=None):
        output = self.siren(model_input,gamma,beta,params=self.get_subdict(params, 'siren'))
        return output

class SingleBVPNetfilmed(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock_filmed(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, gamma,beta,params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone(
        ).detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords, gamma,beta,get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

class FCBlock_filmed(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None
        self.num_hidden_layers =num_hidden_layers

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        # Apply special initialization to first layer, if applicable.
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)

    def forward(self, coords,gamma,beta, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net[0](coords, params=get_subdict(params, 'net.0'))
        output = output*gamma[:32]+beta[:32]
        output = self.net[1](output, params=get_subdict(params, 'net.1'))
        output = output*gamma[32:32*2]+beta[32:2*32]
        output = self.net[2](output, params=get_subdict(params, 'net.2'))
        output = output*gamma[32*2:32*3]+beta[32*2:3*32]
        output = self.net[3](output, params=get_subdict(params, 'net.3'))
        output = output*gamma[32*3:32*4]+beta[32*3:4*32]
        for i in range(4,self.num_hidden_layers+2):
            output = self.net[i](output,params=get_subdict(params, 'net.{}'.format(i)))
        return output

class CNN(MetaModule):
    def __init__(self,resolution, out_features,channel_num):
        super().__init__()
        self.resolution=resolution
        self.out_features = out_features
        self.encoder = MetaSequential(
            MetaConv2d(1, out_features,kernel_size=3, padding=1),
            nn.ReLU(),
            MetaConv2d(out_features, out_features,kernel_size=3, padding=1),
            nn.ReLU(),
            MetaConv2d(out_features, out_features,kernel_size=3, padding=1),
            nn.ReLU(),
            MetaConv2d(out_features, out_features,kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = MetaLinear(self.resolution[0]*self.resolution[1]*channel_num, 1)

    def forward(self,observation,params=None):
        features = self.encoder(observation.squeeze(2).unsqueeze(0).unsqueeze(1), params=self.get_subdict(params, 'encoder'))
        features = self.fc(features.view((features.size(0),256, -1)),params=self.get_subdict(params, 'fc'))
        return features.squeeze(0).squeeze(1)

class LinearFilmGen(MetaModule):
    def __init__(self,in_features=256, out_features=256 * 2,bias=True):
        super().__init__()

        self.fc = MetaLinear(in_features, out_features)

    def forward(self,latent_vector,params=None):
        film = self.fc(latent_vector,params=self.get_subdict(params, 'fc'))
        gamma = film[0:256]
        beta = film[256:]
        return gamma, beta

class CNN_metric(MetaModule):
    def __init__(self,resolution, out_features,channel_num):
        super().__init__()
        self.resolution=resolution
        self.out_features = out_features
        self.encoder = MetaSequential(
            MetaConv2d(channel_num, out_features,kernel_size=3, padding=1),
            nn.LeakyReLU(),
            MetaConv2d(out_features, channel_num,kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # MetaConv2d(out_features, out_features,kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # MetaConv2d(out_features, channel_num,kernel_size=3, padding=1),
            # nn.LeakyReLU()
        )

    def forward(self,observation,params=None):
        features = self.encoder(observation['model_out'].reshape([1,32,32,3]).permute(0,3,1,2))
        return features.permute(0,2,3,1).view(1,-1,3)

# --------------------------------------------
# denoisers tried
# credit to  https://github.com/cszn/DPIR
# --------------------------------------------
# DnCNN
# --------------------------------------------
class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x-n
# --------------------------------------------
# IRCNN
# --------------------------------------------
class IRCNN(MetaModule):
    def __init__(self, in_nc=3, out_nc=3, nc=32):
        super(IRCNN, self).__init__()
        L =[]
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = sequential(*L)

    def forward(self, x,params=None):
        n = self.model(x)
        return x-n

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)