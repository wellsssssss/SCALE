import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models



def dist_contrastive_cls(feat,
                         label,
                         mean=None,
                         covariance=None,
                         ratio=1.0,
                         contrast_temp=100.,
                         num_classes=1000,
                         weight=None,
                         class_weight=None,
                         reduction='mean',
                         avg_factor=None,
                         reg_weight=0,
                         **kwargs):
    assert mean is not None, 'Parameter `mean` required'
    assert covariance is not None, 'Parameter `covariance` required'
    assert not mean.requires_grad
    assert not covariance.requires_grad
    assert feat.requires_grad
    assert not label.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    # feat (N, A) x Ave (A, C)
    temp1 = feat.mm(mean.permute(1, 0).contiguous())
    # feat (N, A)^2 x CoVariance (A, C)
    covariance = covariance * ratio / contrast_temp
    temp2 = 0.5 * feat.pow(2).mm(covariance.permute(1, 0).contiguous())

    logits = temp1 + temp2
    logits = logits / contrast_temp

    # calculate cross-entropy loss
    ce_loss = F.cross_entropy(
        logits,
        label,
        weight=class_weight,
        reduction='none')

    # calculate jcl loss
    jcl_loss = 0.5 * torch.sum(feat.pow(2).mul(covariance[label]), dim=1) / contrast_temp

    loss = ce_loss + jcl_loss

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    if reg_weight > 0.:
        contrast_norm = num_classes * np.log(num_classes)
        loss += reg_weight * proto_reg(feat, mean, contrast_temp, contrast_norm=contrast_norm)

    return loss