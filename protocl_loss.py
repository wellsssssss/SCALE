import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def compute_prototypes(student_model, dataloader):
    
    dataloader = dataloader
    
    student_model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            _, _, projected_features = student_model(images)
            features_list.append(projected_features)
            labels_list.append(labels)
    
    # 只有一个batch，直接使用
    all_features = torch.cat(features_list)
    all_labels = torch.cat(labels_list)
    
    # 计算特征原型
    unique_labels = torch.unique(all_labels)
    prototypes = torch.zeros((len(unique_labels), all_features.size(1))).cuda()
    
    for label in unique_labels:
        prototypes[label] = all_features[all_labels == label].mean(0)
    
    return prototypes

class Protocl(nn.Module):
    def __init__(self,student_model):
        super(Protocl, self).__init__()
        # self.prototypes = prototypes
        self.student_model = student_model
    
    def forward(self, target_images, pseudo_labels,prototypes):
        _, _, target_projected = self.student_model(target_images)
        
        # 计算正样本距离
        pos_proto = prototypes[pseudo_labels]
        pos_dist = F.pairwise_distance(target_projected, pos_proto)
        
        # 计算负样本距离，避免使用原地操作
        all_dists = F.pairwise_distance(target_projected.unsqueeze(1), prototypes.unsqueeze(0), keepdim=True)
        inf_mask = torch.full_like(all_dists, float('inf'))
        neg_dists = torch.where(
            torch.arange(prototypes.size(0), device=all_dists.device) == pseudo_labels.unsqueeze(1).unsqueeze(2),
            inf_mask,
            all_dists
        )
        neg_dists, _ = neg_dists.min(dim=1)
        
        # 计算损失
        loss = pos_dist.sum() + (1.0 / neg_dists.sum())
        
        return loss / len(target_projected)
