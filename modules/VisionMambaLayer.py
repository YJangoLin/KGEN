import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from .mamba_module import *


class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)
    
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x

# class PatchUnEmbed(nn.Module):
#     def __init__(self,basefilter) -> None:
#         super().__init__()
#         self.nc = basefilter
#     def forward(self, x,x_size):
#         B,HW,C = x.shape
#         x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
#         return x

class MSFeatureExtraction(nn.Module):
    def __init__(self,in_channel, base_filter=56, patch_size=1, stride=1, layer_num=8, out_dim=2048):
        super(MSFeatureExtraction, self).__init__()
        self.base_filter = base_filter
        self.stride=stride
        self.patch_size=patch_size
        self.pan_encoder = nn.Sequential(nn.Conv2d(in_channel,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(in_channel,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.embed_dim = base_filter*self.stride*self.patch_size
        self.ms_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.pan_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(layer_num)])
        self.ms_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(layer_num)])
        self.shallow_fusion1 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_fusion2 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        # self.patchunembe = PatchUnEmbed(base_filter)
        self.swap_mamba1 = TokenSwapMamba(self.embed_dim)
        self.swap_mamba2 = TokenSwapMamba(self.embed_dim)
        self.out_swap1 = nn.Sequential(
            nn.Linear(224*224, out_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.out_swap1 = nn.Sequential(
            nn.Linear(224*224, out_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.avg_fnt = nn.AdaptiveAvgPool1d(1)

    def forward(self,ms, pan):
        # ms_bic = F.interpolate(ms,scale_factor=4)
        # pan = F.interpolate(pan,scale_factor=4)
        ms_f = self.ms_encoder(ms)
        # ms_f = ms_bic
        # pan_f = pan
        b,c,h,w = ms_f.shape
        pan_f = self.pan_encoder(pan)
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        residual_pan_f = 0
        ms_f,residual_ms_f = self.ms_feature_extraction([ms_f,residual_ms_f])
        pan_f,residual_pan_f = self.pan_feature_extraction([pan_f,residual_pan_f])
        ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba1(ms_f,pan_f,residual_ms_f,residual_pan_f)
        ms_f,pan_f,residual_ms_f,residual_pan_f = self.swap_mamba2(ms_f,pan_f,residual_ms_f,residual_pan_f)
        ms_f = self.out_swap1(ms_f.transpose(-1, -2)) # 56, 2048
        pan_f = self.out_swap1(pan_f.transpose(-1, -2))
        ms_feat = self.avg_fnt(ms_f.transpose(-1, -2))
        pan_feat = self.avg_fnt(pan_f.transpose(-1, -2))
        return ms_f, pan_f, ms_feat, pan_feat