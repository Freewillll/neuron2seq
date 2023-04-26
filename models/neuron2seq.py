import timm
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import os, sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from datasets.tokenizer import Tokenizer
from utils.util import create_mask
from models.transformer import Transformer


def posemb_sincos_1d(seq, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *seq.shape, seq.device, seq.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)


def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, dim, d, h, w, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(d, device = device),
        torch.arange(w, device = device),
        torch.arange(h, device = device),
        indexing='ij'
    )

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=(3, 3, 3), stride=(1, 1, 1), skip=True):
        super(ResidualBlock, self).__init__()
        padding = tuple((k - 1) // 2 for k in kernel)
        self.skip = skip
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            nn.InstanceNorm3d(outchannel, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(outchannel, affine=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, bias=True),
                nn.InstanceNorm3d(outchannel, affine=True)
            )

    def forward(self, x):
        if self.skip:
            out = self.left(x)
            out = out + self.shortcut(x)
        else:
            out = self.left(x)
        out = F.leaky_relu(out, inplace=True)
        return out
    

class Neu2seq(nn.Module):
    def __init__(self, in_channels, base_channels, down_kernel_list, stride_list,
                 dim, encoder_depth, decoder_depth, heads, dropout, vocab_size, pad_idx):
        super(Neu2seq, self).__init__()
        assert len(down_kernel_list) == len(stride_list)
        self.downs = []
        self.pad_idx = pad_idx

        # the first layer to process the input image
        self.pre_layer = ResidualBlock(in_channels, base_channels, skip=False)

        in_channels = base_channels
        out_channels = 2 * base_channels
        down_filters = []
        self.down_d = 1
        self.down_h = 1
        self.down_w = 1
        for i in range(len(down_kernel_list)):
            down_kernel = down_kernel_list[i]
            stride = stride_list[i]
            self.down_d *= stride[0]
            self.down_w *= stride[1]
            self.down_h *= stride[2]
            down_filters.append((in_channels, out_channels))
            down = ResidualBlock(in_channels, out_channels, kernel=down_kernel, stride=stride)
            self.downs.append(down)
            in_channels = out_channels
            out_channels = out_channels * 2

        out_channels = int(out_channels / 2)        
        
        self.input_proj = nn.Conv3d(out_channels, dim, kernel_size=1)

        self.embedding = nn.Embedding(vocab_size, dim)
        self.transformer = Transformer(dim, heads, encoder_depth, decoder_depth, dropout=dropout)

        # convert layers to nn containers
        self.downs = nn.ModuleList(self.downs)
        self.class_head = nn.Linear(dim, vocab_size)

    def forward(self, img, tgt):
        """
        tgt: shape(B, L, D)
        """

        assert img.ndim == 5
        img = self.pre_layer(img)
        ndown = len(self.downs)
        for i in range(ndown):
            img = self.downs[i](img)

        img = self.input_proj(img)
        pos = posemb_sincos_3d(img)

        tgt_mask, tgt_padding_mask = create_mask(tgt, self.pad_idx)
        tgt_embedding = self.embedding(tgt)

        print(f'tgt_mask: {tgt_mask}')
        print(f'tgt_padding_mask: {tgt_padding_mask}')

        seq_pos = posemb_sincos_1d(tgt_embedding)

        hs = self.transformer(src=img, tgt=tgt_embedding, pos_embed=pos, seq_pos=seq_pos, 
                              tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)[0][0]
        
        out= self.class_head(hs)
        return out
    
    def predict(self, img, tgt, args):
        length = tgt.size(1)
        padding = torch.ones(tgt.size(0), args.max_seq_len-length-1).fill_(self.pad_idx).long().to(tgt.device)
        tgt = torch.cat([tgt, padding], dim=1)

        print(f'tgt: {tgt}')

        assert img.ndim == 5
        img = self.pre_layer(img)
        ndown = len(self.downs)
        for i in range(ndown):
            img = self.downs[i](img)

        img = self.input_proj(img)
        pos = posemb_sincos_3d(img)

        tgt_mask, tgt_padding_mask = create_mask(tgt, self.pad_idx)
        tgt_embedding = self.embedding(tgt)
        
        seq_pos = posemb_sincos_1d(tgt_embedding)
        
        hs = self.transformer(src=img, tgt=tgt_embedding, pos_embed=pos, seq_pos=seq_pos, 
                              tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)[0][0]
        
        out= self.class_head(hs)

        test = torch.argmax(out, dim=-1)
        print(f'test: {test}')

        return out[:, length-1, :]
    

if __name__ == '__main__':
    from torchinfo import summary
    import argparse

    parser = argparse.ArgumentParser(
        description='test')
    args = parser.parse_args()
    args.down_kernel_list = [[1,3,3], [3,3,3], [3,3,3], [3,3,3]]
    args.stride_list = [[1,2,2], [2,2,2], [2,2,2], [2,2,2]]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = torch.randn(2, 1, 32, 64, 64).to(args.device)
    tgt = torch.rand(2, 10) * 64
    tgt = tgt.type(torch.int64).to(args.device)

    tokenizer = Tokenizer(num_classes=4, num_bins=64, depth=32,
                          width=64, height=64, max_len=3)
    args.pad_idx = tokenizer.PAD_code

    model = Neu2seq(in_channels=1, base_channels=24, down_kernel_list=args.down_kernel_list, 
                    stride_list=args.stride_list, dim=256, encoder_depth=6, decoder_depth=6,
                    heads=8, dropout=0.1, vocab_size=tokenizer.vocab_size, pad_idx=args.pad_idx).to(args.device)
    print(model)

    outputs = model(img, tgt)
    # print(outputs)
    summary(model, input_data=(img, tgt))
