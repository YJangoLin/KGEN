import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN, clones, MultiHeadedAttention
# from modules.base_mam import BaseMAM
from modules.visual_extractor import VisualExtractor
from modules.ImgFeatureAlignModule import ImageFratureAlign
from modules.KGE_index import KLRetrieval
from modules.KGEFuseblock import KGEFuseBlock
from mamba_ssm.modules.mamba_simple import Mamba
# from modules.mamba_module import LayerNorm
# from modules.VisionMambaLayer import MSFeatureExtraction

class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.useMAM = args.useVTFCM
        # if args.visual_extractor == 'mamba': self.visual_extractor = MSFeatureExtraction(in_channel=3,layer_num=args.mam_layer)
        # else: self.visual_extractor = VisualExtractor(args)
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        self.AM = args.AM
        # self.visualMatrix = torch.randn(56, 56)
        if args.AM == "VFAM":
            self.ImageFA = ImageFratureAlign(in_linear_feature=[2048, 1024], out_linear_feature=[1024, args.d_model], d_model=self.args.d_vf)
        elif args.AM == "SA":
            self.ImageFA = MultiHeadedAttention(h=8, d_model=2048)
        elif args.AM == "CM":
            self.ImageFA = Mamba(d_model=2048)
        else:
            self.ImageFA = None
        # self.linear = nn.Identity()
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
            # self.linear = nn.Linear(50176, 2048)
        if args.useKG:
            self.KGIndex = KLRetrieval(output_dim=12)
        else:
            self.KGIndex = nn.Identity()
            # self.KGFuseBlock = nn.Identity()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, cls=None,  mode='train', update_opts={}):
        # if self.args.visual_extractor == 'mamba':
        #     att_feats_0, att_feats_1,fc_feats_0, fc_feats_1 = self.visual_extractor(images[:, 0], images[:, 1])
        # else:
        if self.args.isSingle:
            att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0]) # att_feats_0: (16, 49, 2048)
            att_feats_1, fc_feats_1 = att_feats_0, fc_feats_0
        else:
            att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0]) # att_feats_0: (16, 49, 2048)
            att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1]) # att_feats_0: (16, 49, 2048)
        if self.AM == "VFAM":
            att_feats = self.ImageFA(att_feats_0, att_feats_1)
        elif self.AM == "SA":
            att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
            att_feats = self.ImageFA(att_feats, att_feats, att_feats)
        elif self.AM == "CM":
            att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
            att_feats = self.ImageFA(att_feats)
        else:
            att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1) # (B, 2048)
        clsLoss = 0.0
        # 嵌入知识图
        if self.args.useKG:
            att_feats, clsLoss = self.KGIndex(fc_feats, att_feats, cls)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output, clsLoss
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, cls=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if self.AM == "VFAM":
            att_feats = self.ImageFA(att_feats, att_feats)
        elif self.AM == "SA":
            # att_feats = torch.cat((att_feats, att_feats), dim=1)
            # att_feats = att_feats.transpose(1, 2)
            # att_feats = self.linear(att_feats)
            att_feats = self.ImageFA(att_feats, att_feats, att_feats)
        elif self.AM == "CM":
            # att_feats = torch.cat((att_feats, att_feats), dim=1)
            att_feats = self.ImageFA(att_feats)
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1) # (B, 2048)
        clsLoss = 0.0
        # 嵌入知识图
        if self.args.useKG:
            att_feats, clsLoss = self.KGIndex(fc_feats, att_feats, cls)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output, clsLoss
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
