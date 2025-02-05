import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# Positional encoding (convert integer 't' into a vector that contains a positional information)
def _pos_encoding(t, output_dim, device = 'cpu'):
    D = output_dim # positional encoding vector dimension
    v = torch.zeros(D, device = device)
    i = torch.arange(0, D, device = device)

    div = 10000 ** (i / D)

    v[0::2] = torch.sin(t / div[0::2]) # even
    v[1::2] = torch.cos(t / div[1::2]) # odds

    return v

# POS Encoding for Batch 
def pos_encoding(ts, output_dim, device = 'cpu'):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device = device) # (N,D)

    for i in range(batch_size): v[i] = _pos_encoding(ts[i], output_dim, device = device)

    return v #(N,D)

# Convblock
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_ch):
        super().__init__()

        # Conv layers
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(out_ch)
        )
        
        # Position embedding linear translation
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_ch, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    # Input?: x (pic data) -> (N,C,H,W) and Timestep Embedding Vector (N,D)
    def forward(self, x, v):

        N, C, _, _ = x.shape
        v = self.mlp(v) # v -> (N,C)
        v = v.view(N,C,1,1)
        x = x + v # Adding time information for every Convblock
        y = self.convs(x)

        return y

# UNet (Neural Net for Diffusion Model)
class UNet(nn.Module):
    def __init__(self, in_ch = 1, time_emb_dim = 100):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Conv layers
        self.down1 = ConvBlock(in_ch, 64, time_emb_dim)
        self.down2 = ConvBlock(64, 128, time_emb_dim)
        self.bot1 = ConvBlock(128, 256, time_emb_dim)
        self.up2 = ConvBlock(128+256, 128, time_emb_dim)
        self.up1 = ConvBlock(64+128, 64, time_emb_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        # Downsample & Upsample
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    def forward(self, x, timesteps):

        v = pos_encoding(timesteps, self.time_emb_dim, x.device) # (N,time_emb_dim)

        # Down
        x1 = self.down1(x,v)
        x = self.downsample(x1)
        x2 = self.down2(x,v)
        x = self.downsample(x2)

        # Bot
        x = self.bot1(x, v)

        # Up
        x = self.upsample(x)
        x = torch.cat([x, x2], dim = 1) # Skip connection 1
        x = self.up2(x,v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim = 1) # Skip connection 2
        x = self.up1(x,v)
        x = self.out(x)

        return x
    






