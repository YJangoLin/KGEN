import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class linearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, channel) -> None:
        super(linearLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activate  = nn.ReLU()
        self.norm = nn.BatchNorm1d(num_features=channel)
    
    def forward(self, src):
        return self.norm(self.activate(self.linear(src)))


class FusionBlock(nn.Module):
    def __init__(self, topk=20) -> None:
        super(FusionBlock, self).__init__()
        self.topk = topk
        self.sm = nn.Softmax(dim=1)

    def forward(self, src1, src2, memoryMartix):
        # (1, 49, 2048)
        n = memoryMartix.shape[-1]
        memoryMartix = self.sm(memoryMartix)
        vtopk1, ind1 = torch.topk(memoryMartix, self.topk, 1) #vtopk:(49, 20) ind:(49, 20)
        src = torch.zeros_like(src1) # (1, 49, 20, 2048)
        for i in range(n):
            src[:, i, :] = (src2[:, ind1[i,:], :] * vtopk1[i, :].unsqueeze(0).unsqueeze(-1)).sum(dim=1) # (1, 20, 2048)
        # (1, 49, 20, 1)
        # ind1 = ind1.unsqueeze(0).unsqueeze(-1).expand(src1.shape[0], ind1.shape[0], ind1.shape[1], src1.shape[2], src1.shape[3]) # (49, 20)
        # src2topk = src2.unsqueeze(2).expand(src2.shape[0], src2.shape[1], n, src2.shape[2], src2.shape[3])
        # src1repose = ((src2topk.gather(src2topk, ind1) * vtopk1.unsqueeze(0).unsqueeze(-1)).sum(dim = -2)).squeeze() # (n, 49, 2048)
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

class ImageFratureAlign(nn.Module):
    def __init__(self, in_linear_feature, out_linear_feature, d_model=2048, topk=20, mem_dim=49) -> None:
        super(ImageFratureAlign, self).__init__()
        mlp = nn.Sequential(*[linearLayer(in_feature, out_feature, channel=mem_dim)   for in_feature, out_feature in zip(in_linear_feature, out_linear_feature)])
        self.sm1 = nn.Softmax(dim=1)
        self.sm2 = nn.Softmax(dim=0)
        self.topk = topk
        # ? 这样直接用线性层会不会丢失位置信息
        self.mlps = clones(mlp, 2)
        self.linear1 = nn.Linear(mem_dim, mem_dim)
        self.act = nn.Tanh()
        self.fusionBlock = FusionBlock(topk)
        # ? 自注意力能不能换成通道注意力ECA
        # self.atten = selfAttention(head_num, in_linear_feature[0], d_model)
        self.norm = LayerNorm(d_model)
        self.visualMatrix = nn.Parameter(torch.normal(mean=1,std=0.05,size=(mem_dim,mem_dim)), requires_grad=False).cuda()
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier初始化权重
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

    def forward(self, src1, src2):
        # ? batch 考虑不考虑，暂时不考虑
        n = self.visualMatrix.shape[-1]
        assert src1.shape[-2] == n and src2.shape[-2] == n
        f1 = self.mlps[0](src1)
        f2 = self.mlps[1](src2) # （49, 256）
        scores = torch.matmul(f1, f2.transpose( 1, 2)).sum(dim=0) # (n, 49, 49) --> (49, 49)
        scores = self.act(self.linear1(scores))
        self.visualMatrix = self.visualMatrix + scores.detach()
        src1 = self.fusionBlock(src1, src2, self.visualMatrix)
        src2 = self.fusionBlock(src2, src1, self.visualMatrix.transpose(0, 1))
        #? 会不会不计算src2， 直接输入src1会好一些。然后用一个自注意力机制
        src = torch.cat([src1, src2], dim=1)
        # 添加位置信息

        return self.norm(src)

# if __name__ == '__main__':
#     data1 = torch.randn(1, 2048, 49)
#     data2 = torch.randn(1, 2048, 49)
#     visualMatrix = torch.randn(49, 49)
#     imagefA = ImageFratureAlign(in_linear_feature=[2048, 1024], out_linear_feature=[1024, 256], head_num=8, d_model=256)
#     imagefA(data1.transpose(-1, -2), data2.transpose(-1, -2), visualMatrix)
    



