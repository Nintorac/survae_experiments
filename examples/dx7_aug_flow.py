#%%
import math
import torch

# Data
from survae.data.loaders.image import MNIST

# Model
import torch.nn as nn
from survae.flows import Flow
from survae.distributions import StandardNormal, StandardUniform
from survae.transforms import AffineCouplingBijection, ActNormBijection1d as ActNormBijection2d, Conv1x11d as Conv1x1
from survae.transforms import UniformDequantization, Augment, Squeeze1d as Squeeze2d, Slice
from survae.nn.layers import ElementwiseParams1d as ElementwiseParams2d
from survae.nn.nets import DenseNet

from agoge.data_handler import DataHandler
from neuralDX7.datasets import SingleVoiceLMDBDataset

# Optim
from torch.optim import Adam

# Plot
import torchvision.utils as vutils

############
## Device ##
############

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##########
## Data ##
##########

data = DataHandler(SingleVoiceLMDBDataset, {
            'keys_file': 'unique_voice_keys.npy',
            'data_file': 'dx7-data.lmdb',
            'root':'~/agoge/artifacts/data',
            'data_size': 1.
        }, {'shuffle': True, 'batch_size': 32}
        )

train_loader, test_loader, _ = data.loaders

###########
## Model ##
###########

class Net(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=64, n_heads=8, layers=2, **kwargs):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=mid_channels, nhead=n_heads)
        self.net = transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self._in = nn.Linear(in_channels, mid_channels)
        self._out = nn.Linear(mid_channels, out_channels)

    def forward(self, x):


        batch_dim, channel_dim, sequence_dim = range(3)

        x = x.permute(batch_dim, sequence_dim, channel_dim)
        batch_dim, channel_dim, sequence_dim = batch_dim, sequence_dim, channel_dim

        x = self._out(self.net(self._in(x)))

        x = x.permute(batch_dim, channel_dim, sequence_dim)

        return x

def net(channels):
  return nn.Sequential(
#       DenseNet(in_channels=channels//2,
#                                 out_channels=channels,
#                                 num_blocks=1,
#                                 mid_channels=64,
#                                 depth=8,
#                                 growth=16,
#                                 dropout=0.0,
#                                 gated_conv=True,
#                                 zero_init=True),
                        # nn.Conv1d(channels//2, channels, 1, bias=False),
                        Net(    
                            in_channels=channels//2, 
                            out_channels=channels,
                            mid_channels=64
                            ),
                        ElementwiseParams2d(2))

base_channels=1
n_items = 256

def perm_norm_bi(channels):
    return AffineCouplingBijection(net(channels)), \
            ActNormBijection2d(channels),  \
            Conv1x1(channels)


def reduction_layer(channels, items):
    return [
        *perm_norm_bi(channels),
        *perm_norm_bi(channels),
        *perm_norm_bi(channels),
        Squeeze2d(4), 
        Slice(StandardNormal((channels*2, items)), num_keep=channels*2),
    ]


model = Flow(base_dist=StandardNormal((base_channels*(2**5), n_items//(4**4))),
             transforms=[
                UniformDequantization(num_bits=8),
                Augment(StandardUniform((base_channels*1, n_items)), x_size=base_channels),
                *reduction_layer(base_channels*(2**1), n_items//(4**1)),
                *reduction_layer(base_channels*(2**2), n_items//(4**2)),
                *reduction_layer(base_channels*(2**3), n_items//(4**3)),
                *reduction_layer(base_channels*(2**4), n_items//(4**4)),
                # *reduction_layer(base_channels*(2**5), n_items//(4**4)),
                *perm_norm_bi(base_channels*(2**5))
                
                # AffineCouplingBijection(net(base_channels*2)), ActNormBijection2d(base_channels*2), Conv1x1(base_channels*2),
                # AffineCouplingBijection(net(base_channels*2)), ActNormBijection2d(base_channels*2), Conv1x1(base_channels*2),
                # AffineCouplingBijection(net(base_channels*2)), ActNormBijection2d(base_channels*2), Conv1x1(base_channels*2),
                # Squeeze2d(), Slice(StandardNormal((base_channels*2, n_items//4)), num_keep=base_channels*2),
                # AffineCouplingBijection(net(base_channels*2)), ActNormBijection2d(base_channels*2), Conv1x1(base_channels*2),
                # AffineCouplingBijection(net(base_channels*2)), ActNormBijection2d(base_channels*2), Conv1x1(base_channels*2),
                # AffineCouplingBijection(net(base_channels*2)), ActNormBijection2d(base_channels*2), Conv1x1(base_channels*2),
                
             ]).to(device)

x = next(iter(train_loader))

x = x['X']
x = x.unsqueeze(1)
x = torch.cat([x, torch.zeros_like(x[...,[0]*101])], dim=-1)
print(model.log_prob(x))

print('back')
print('back')
print('back')
print('back')
print('back')
print('back')
print('back')
print('back')
model.sample(64).shape
#%%
###########
## Optim ##
###########

optimizer = Adam(model.parameters(), lr=1e-3)

###########
## Train ##
###########

print('Training...')
for epoch in range(1000):
    l = [0.0]
    for i, x in enumerate(train_loader):
        x = x['X']
        x = x.unsqueeze(1)
        x = torch.cat([x, torch.zeros_like(x[...,[0]*101])], dim=-1)
        # x = (torch.rand_like(x.squeeze(-1).float()) * 56).long()
        optimizer.zero_grad()
        loss = -model.log_prob(x.to(device)).sum() / (math.log(2) * x.numel())
        loss.backward()
        optimizer.step()
        l += [loss.detach().cpu().item()]
        print('Epoch: {}/{}, Iter: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, 10, i+1, len(train_loader), l[-1]), end='\r')
    print('')

##########
## Test ##
##########

print('Testing...')
with torch.no_grad():
    l = 0.0
    for i, x in enumerate(test_loader):

        x = x['X']
        x = x.unsqueeze(1)
        x = torch.cat([x, torch.zeros_like(x[...,[0]*101])], dim=-1)
        loss = -model.log_prob(x.to(device)).sum() / (math.log(2) * x.numel())
        l += loss.detach().cpu().item()
        print('Iter: {}/{}, Bits/dim: {:.3f}'.format(i+1, len(test_loader), l/(i+1)), end='\r')
    print('')

############
## Sample ##
############

print('Sampling...')
img = (data.test.data[:64][:,None][:,[0]*3])#.permute([0,3,1,2])
samples = model.sample(64)

vutils.save_image(img.cpu().float()/255, fp='cifar10_data.png', nrow=8)
vutils.save_image(samples.cpu().float()/255, fp='cifar10_aug_flow.png', nrow=8)

# %%
