from typing import Tuple

import torch
import torch.nn as nn

from ..blocks.utils import get_padding


# Source: https://github.com/luuuyi/CBAM.PyTorch
class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.fc1 = nn.Linear(in_planes * 3, in_planes // ratio)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_planes // ratio, in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : [bs, C, T]
        avg_out = torch.mean(x, dim=2, keepdim=False)
        std_out = torch.std(x, dim=2, keepdim=False).clamp(min=1e-10,max=torch.finfo(x.dtype).max)
        max_out, _ = torch.max(x, dim=2, keepdim=False)

        x = torch.cat([avg_out, std_out, max_out], dim=1)

        out = self.fc2(self.relu1(self.fc1(x)))
        out = out.unsqueeze(-1)
        return self.sigmoid(out).clamp(min=1e-6,max=torch.finfo(x.dtype).max)

# Source: https://github.com/luuuyi/CBAM.PyTorch
class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()

        assert (kernel_size % 2) == 1  # odd
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : [bs, C, T]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        std_out = torch.std(x, dim=1, keepdim=True).clamp(min=1e-10,max=torch.finfo(x.dtype).max)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, std_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x).clamp(min=1e-6,max=torch.finfo(x.dtype).max)

# Attention modules from: https://github.com/lRomul/argus-freesound/blob/master/src/models/aux_skip_attention.py
class ConvolutionalBlockAttentionModule1D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule1D, self).__init__()
        self.ca = ChannelAttention1D(in_planes, ratio)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, input):
        out = self.ca(input) * input
        out = self.sa(out) * out
        return out

class ResidualBlock(nn.Module):
    def __init__(self,
                 nb_filts : Tuple[int,int],
                 first : bool = False,
                 pool_stride : int = 1,
                 add_attention : bool = True,
                 dropout : float = 0.0
                 ):
        super(ResidualBlock, self).__init__()
        self.pool_stride = pool_stride
        self.add_attention = add_attention
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            # self.dropout = nn.ConstantPad2d(0, 0)
            self.dropout = lambda x : x
        self.first = first

        if self.add_attention:
            self.attention_block = ConvolutionalBlockAttentionModule1D(in_planes=nb_filts[0])

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.01,inplace=True)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)
        else:
            self.downsample = False
        if self.pool_stride > 1:
            self.mp = nn.MaxPool1d(self.pool_stride)

    def forward(self, x):
        identity = x
        if self.add_attention:
            x = self.attention_block(x)
        if not self.first:
            x = self.bn1(x)
            x = self.lrelu_keras(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.lrelu_keras(x)
        x = self.conv2(x)

        if self.downsample:
            identity = self.conv_downsample(identity)

        x += identity
        if self.pool_stride > 1:
            x = self.mp(x)
        
        # Apply spatial dropout [b,C,T]
        x = x[:,:,None,:]
        x = self.dropout(x)
        x = x[:,:,0,:]
        return x

def make_conv1d(in_planes,out_planes,ks=3,stride=1):
    conv = nn.Conv1d(in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=ks,
                    padding=get_padding(ks),
                    stride=stride)
    bn = nn.BatchNorm1d(num_features=out_planes)
    relu = nn.LeakyReLU()
    return nn.Sequential(*[conv,bn,relu])
    
def normalize(signal,axis=-1,keepdims=True):
    max_value = signal.abs().max(axis=axis,keepdims=keepdims).values
    max_value = max_value.clip(1e-8,None)
    return signal / max_value

class RawNet(nn.Module):
    def __init__(self,
                normalize_input = True,
                init_conv_params = dict(
                    in_channels=1,
                    out_channels=96,
                    stride=3,
                    kernel_size=3,
                    padding=0,
                ),
                block_dropout = 0.0,
                block_setup = [
                     (96, 128, True, 1),
                     (128, 128, True, 3),
                     (128, 160, True, 1),
                    
                     (160, 160, True, 3),
                     (160, 192, True, 1), 
                    
                     (192, 192, True, 3),
                     (192, 256, True, 1), 

                     (256, 256, True, 3),
                     (256, 288, True, 1), 

                     (288, 288, True, 3),
                     (288, 288, True, 1), 
                ],
                ):
        super(RawNet, self).__init__()
        self.normalize_input=normalize_input
        self.conv1 = nn.Conv1d(**init_conv_params)
        self.bn1 = nn.BatchNorm1d(num_features=init_conv_params["out_channels"])
        self.relu = nn.ReLU(inplace=True)

        self.resnet = nn.Sequential(*[ResidualBlock(
                    nb_filts=(init_conv_params["out_channels"], block_setup[0][1]),
                    first=True, pool_stride=block_setup[0][3], dropout=block_dropout,
                    add_attention=False)] +
                [ResidualBlock(
                    nb_filts=(filts0, filts1), dropout=block_dropout,
                    first=False, pool_stride=pool_stride,
                    add_attention=add_attention_blocks)
                        for filts0, filts1, add_attention_blocks, pool_stride in block_setup[1:]
                 ])
        
    def forward(self, waveform):
        # waveform : [bs, 1, T]
        if self.normalize_input:
            waveform = normalize(waveform)
        
        waveform = waveform.float()

        x = self.conv1(waveform)
        x = self.bn1(x)
        x = self.relu(x)

        for block in self.resnet:
            x = block(x)
        return x
