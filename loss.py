import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, prototypes=None, covariances=None):
        """Compute loss for model. 
        
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            prototypes: class prototypes of shape [n_classes, feature_dim]
            covariances: class covariance matrices of shape [n_classes, feature_dim, feature_dim]
        Returns:
            A loss scalar.
        """
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        similarity_matrix = torch.matmul(anchor_feature, contrast_feature.T)
        if prototypes is not None: # ProtoCL
            protoypes = F.normalize(prototypes, dim=-1)
            similarity_proto = torch.matmul(anchor_feature, protoypes.T)
            logits = torch.cat([similarity_matrix, similarity_proto], dim=-1) 
        elif covariances is not None: # DistCL
            dis_matrix = torch.zeros(batch_size, batch_size).to(features.device)
            for i in range(batch_size):
                for j in range(batch_size):
                    dis_matrix[i,j] = torch.matmul(anchor_feature[i], torch.matmul(covariances[labels[i]], anchor_feature[j]))
            logits = similarity_matrix + dis_matrix / (2 * self.temperature**2)
        else: # Without CL
            logits = similarity_matrix
        
        logits = logits / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True)+1e-12)
    
    # compute mean of log-likelihood over positive
        epsilon = 1e-12
        mask = mask.unsqueeze(-1)  # 在最后一维上添加一个新的维度
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + epsilon)

    # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
    
        return loss
        

class RegularizationLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(RegularizationLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, prototypes):   
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        
        similarity_matrix = torch.matmul(features, prototypes.T)       
        logits = similarity_matrix / self.temperature
        
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-12)
        
        loss = -log_prob.mean(0).sum()
        
        return loss
    
def compute_ssl_loss(teacher_logits, target_logits, threshold=0.9):
    # 计算伪标签的概率和置信度
    pseudo_probs = F.softmax(teacher_logits, dim=1)
    max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
    
    # 选择置信度高于阈值的样本
    mask = max_probs > threshold
    selected_target_logits = target_logits[mask]
    selected_pseudo_labels = pseudo_labels[mask]
    
    # 如果有足够的样本超过阈值，则计算损失
    if len(selected_target_logits) > 0:
        ssl_loss = F.cross_entropy(selected_target_logits, selected_pseudo_labels)
    else:
        # 如果没有任何样本满足条件，返回一个零损失或者避免计算损失
        ssl_loss = torch.tensor(0.0).cuda()
    
    return ssl_loss