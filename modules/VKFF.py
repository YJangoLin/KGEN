import torch
import torch.nn as nn
from modules.base_cmn import MultiHeadedAttention
from mamba_ssm.modules.mamba_simple import Mamba

class VKFF(nn.Module):
    def __init__(self, d_m=2048):
        super(VKFF, self).__init__()
        self.d_m = d_m
        self.ImageFA = MultiHeadedAttention(h=8, d_model=2048)
        self.cross_atten = Mamba(
            d_model=d_m, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,   # Local convolution width
            expand=2,
        )

    def forward(self,imageFeature, kf):
        att_feats = self.ImageFA(imageFeature, imageFeature, imageFeature)
        src = self.cross_atten(imageFeature, extra_emb=kf)
        pass
