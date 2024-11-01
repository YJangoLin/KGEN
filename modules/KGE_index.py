# 经过ResNet101以后得数据，avg_feats为（batch_size， 2048）, patch_feats为（batch_size, 49, 2048）
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from .muti_attention import MultiHeadedAttention
from .KGEFuseblock import KGEFuseBlock
# from torch_geometric.nn import GCNConv

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=1024, hidden_dim2=256, output_dim=12):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmod(x)


class KLRetrieval(nn.Module):
    def __init__(self, output_dim=12, entity_embedding_file="data/KGE/entity_embedding.npy", relation_embedding_file="data/KGE/relation_embedding.npy", dataFile="data/data.csv"):
        super(KLRetrieval, self).__init__()
        self.df = pd.read_csv(dataFile, usecols=["le","re","rela","cls"])
        self.entitysEmbed = np.load(entity_embedding_file)
        self.relaEmbed = np.load(relation_embedding_file)
        self.mlp = MLP(input_dim=4096, output_dim=output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.adppool = nn.AdaptiveAvgPool1d(256)
        self.atten = MultiHeadedAttention(h=8, d_model=2048, dropout=0.1)
        self.kgef = KGEFuseBlock()
        
    
    def forward(self, x, imageFeature, clsLabel):
        logits  = self.mlp(x) # (6, 12)
        # cls = torch.argmax(x, dim=1)
        # 计算损失
        clsLoss = self.criterion(logits,clsLabel.long())
        # 索引
        cls = torch.argmax(logits, dim=1)
        src = torch.zeros_like(imageFeature)
        # 实体关系使用attention或者menda进行融合，然后输入到图像特征中
        # 把两个实体和对应关系进行融合，得到的特征进行
        for index, c in enumerate(cls):
            clsDf = self.df[self.df["cls"]==c.item()]
            lebembed = self.entitysEmbed[clsDf["le"].tolist()]
            rebembed = self.entitysEmbed[clsDf["re"].tolist()]
            embed = torch.from_numpy(np.concatenate([lebembed, rebembed], axis=0)).to(x.device)
            relabembed = torch.from_numpy(self.relaEmbed[clsDf["rela"].tolist()]).to(x.device)
            kf = (self.atten(embed.unsqueeze(0), relabembed.unsqueeze(0), relabembed.unsqueeze(0)) + embed.unsqueeze(0)).transpose(1, 2) # (1, dim, n_k)
            src[index,:, :] = self.kgef(imageFeature[index,:, :], kf)
        return imageFeature+src, clsLoss
    

if __name__ == '__main__':
    x = torch.randn(6, 2048)
    clsLabel = torch.tensor([1,2,0,2,1,7])
    KLR = KLRetrieval()
    KLR(x, clsLabel)




