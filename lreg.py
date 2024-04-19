import torch
import torch.nn as nn
import torch.nn.functional as F

class LregLoss(nn.Module):
    def __init__(self, num_classes, temp=0.1):
        super(LregLoss, self).__init__()
        self.num_classes = num_classes
        self.temp = temp
    
    def forward(self, domain_features, dist_means):
        # 计算当前批次的全局特征均值
        batch_mean = domain_features.mean(dim=0)
        
        # 计算batch_mean与每个类别均值的相似度
        similarities = torch.exp((batch_mean * dist_means).sum(1) / self.temp)
        
        # 计算Lreg损失
        loss = -torch.log(similarities / similarities.sum()).sum() / (self.num_classes * torch.log(torch.tensor(self.num_classes)))
        
        return loss