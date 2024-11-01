import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from .muti_attention import MultiHeadedAttention
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x), h, w

class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,ms,ms_resi,pan):
        ms_resi = ms+ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms),extra_emb=self.norm2(pan))
        B,HW,C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, 128, 128)
        ms =  (self.dwconv(ms)+ms).flatten(2).transpose(1, 2)
        return ms,ms_resi


class KGEFuseBlock(nn.Module):
    def __init__(self):
        super(KGEFuseBlock, self).__init__()
        # self.linear1 = nn.Linear(KG_input_dim, KG_output_dim)
        # self.atten = MultiHeadedAttention(h=8, d_model=2048)
        self.relu = nn.ReLU()
        # self.weightKLMatrix = nn.Parameter(torch.normal(mean=1,std=0.05,size=(batch_size, 98, 2048)), requires_grad=False).cuda()
        self.atten = Mamba(
            d_model=98, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,   # Local convolution width
            expand=2,
        )
        self.cross_atten = Mamba(
            d_model=2048, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,   # Local convolution width
            expand=2,
        )
        self.tanh1 = nn.Tanh()
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,imageFeature, kf):
        B = kf.shape[0]
        kf = kf.transpose(1, 2) # (b, 2048, n_k) -> (b, n_k, 2048)
        imageFeature = imageFeature.unsqueeze(0)
        IKScore = torch.matmul(kf, imageFeature.transpose(1, 2)) # # (b, n_k, 98)
        IKScore = self.atten(IKScore)
        # !:补充cross-attention
        # IKScore = self.pool(IKScore) # (b, n_k)
        IKScore = self.softmax(self.pool(IKScore).squeeze(-1)) # pool:(b, n_k)   softmax: (b, n_k)
        # self.weightKLMatrix += self.tanh1(IKScore)
        # 选择出最大的98个
        vtopk1, ind1 = torch.topk(IKScore, 98, 1) # ind1
        src = torch.zeros(B, 98, 2048).to(IKScore.device)
        for i in range(B):
            src[i, :, :] = kf[i, ind1[i,:], :] * vtopk1[i, :].unsqueeze(0).unsqueeze(-1) # (1, 98, 2048) * (1, 98, 1) = (1, 98, 2048)
        # x = self.relu(self.linear1(x))
        # x = self.atten(imageFeature, x, x)
        src = self.cross_atten(imageFeature, extra_emb=src)
        return src