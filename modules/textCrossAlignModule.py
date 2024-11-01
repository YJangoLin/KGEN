
import torch
import copy
import torch.nn as nn
import math
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class linearLayer(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(linearLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activate  = nn.ReLU()
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, src):
        return self.norm(self.activate(self.linear(src)))


class FusionBlock(nn.Module):
    def __init__(self, topk=20) -> None:
        super(FusionBlock, self).__init__()
        self.topk = topk
        self.sm = nn.Softmax(dim=1)

    def forward(self, src1, memoryMartix):
        # report 的长度不一样有可能小于topk所以不能这样
        n = src1.shape[-2] # (B, 98, 512)
        memoryMartix = self.sm(memoryMartix)
        vtopk1, ind1 = torch.topk(memoryMartix, self.topk, 1) #vtopk:(98, 32) ind:(98, 32)  #vtopk:(60, 32) ind:(60, 32)
        src = torch.zeros_like(src1) # (12, 59, 512)
        for i in range(n):
            src[:, i, :] = (memoryMartix[:, ind1[i,:], :] * vtopk1[i, :].unsqueeze(0).unsqueeze(-1)).sum(dim=1) # (1, 32, 1)
        return src + src1


class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class TextToImageCrossAlignModule(nn.Module):
    def __init__(self, in_linear_feature, out_linear_feature, multi_atten, d_model=2048, topk=20, mem_dim=98, maxLenReport=80, dropout=0.1, embed_dim=98) -> None:
        assert topk<mem_dim and topk < maxLenReport, print("topk>dim")
        super(TextToImageCrossAlignModule, self).__init__()
        mlp = nn.Sequential(*[linearLayer(in_feature, out_feature)   for in_feature, out_feature in zip(in_linear_feature, out_linear_feature)])
        self.sm1 = nn.Softmax(dim=1)
        self.sm2 = nn.Softmax(dim=2)
        self.topk = topk
        # ? 这样直接用线性层会不会丢失位置信息
        self.mlps = clones(mlp, 2)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.act = nn.Sigmoid()
        self.act1 = nn.Tanh()
        self.fusionBlock = FusionBlock(topk)
        # ? 自注意力能不能换成通道注意力ECA
        self.norm = LayerNorm(d_model)
        self.src_atten = multi_atten
        self.rep_atten = copy.deepcopy(multi_atten)
        # self.atten = copy.deepcopy(multi_atten)
        self.dropout = nn.Dropout(dropout)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier初始化权重
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

    def forward(self, ImageSrc, reportSrc, visualMatrix): # 
        # ? batch 考虑不考虑，暂时不考虑
        B, rrc, _ = reportSrc.shape
        ic, rc = visualMatrix.shape
        assert ImageSrc.shape[-2] == ic and reportSrc.shape[-2] <= rc
        f1 = self.dropout(self.mlps[0](ImageSrc)) # (6, 98, 512)
        f2 = self.dropout(self.mlps[1](reportSrc)) # (6, varlen, 512)
        scores = torch.matmul(f2, f1.transpose(1, 2)) # (n, varlen, 98)
        scores = self.act(self.linear(scores)).transpose(1, 2) # (n, 98, varlen)
        imageScores = self.sm2(scores) 
        reportScores = self.sm1(scores)
        reportMatrix = torch.matmul(imageScores.transpose(1, 2), ImageSrc)
        visualMatrixExpand = visualMatrix.unsqueeze(0).expand(B, visualMatrix.shape[0], visualMatrix.shape[1]) + self.act1(torch.matmul(reportScores, reportSrc))
        isrc = self.src_atten(visualMatrixExpand, reportMatrix, reportMatrix)
        rsrc = self.rep_atten(reportMatrix, visualMatrixExpand, visualMatrixExpand)
        return ImageSrc + isrc, reportSrc + rsrc
