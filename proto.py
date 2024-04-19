import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, n_classes, feature_dim, memory_size=1024):
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        # Initialize memory bank
        self.memory = torch.zeros(memory_size, feature_dim).cuda()
        self.targets = torch.zeros(memory_size).long().cuda()
        self.ptr = 0
        
    def update(self, features, targets):
        batch_size = features.shape[0]
        
        # Replace old features and targets with new ones
        if self.ptr + batch_size > self.memory_size:
            remainder = self.memory_size - self.ptr
            self.memory[self.ptr:] = features[:remainder].cuda()
            self.targets[self.ptr:] = targets[:remainder].cuda()
            self.ptr = batch_size - remainder
            self.memory[:self.ptr] = features[remainder:].cuda()
            self.targets[:self.ptr] = targets[remainder:].cuda()
        else:
            self.memory[self.ptr:self.ptr+batch_size] = features.cuda()
            self.targets[self.ptr:self.ptr+batch_size] = targets.cuda()
            self.ptr += batch_size
            
    def sample(self, batch_size, targets):
        neg_indices = torch.randint(0, self.memory_size, (batch_size,))
        neg_features = self.memory[neg_indices]
        neg_targets = self.targets[neg_indices]
    
        mask = neg_targets == torch.unsqueeze(targets, 1)
        num_tries = 0
        max_tries = 10
    
        while mask.any() and num_tries < max_tries:
            new_neg_indices = torch.randint(0, self.memory_size, (batch_size,))
            new_neg_features = self.memory[new_neg_indices]
            new_neg_targets = self.targets[new_neg_indices]
        
            mask = new_neg_targets == torch.unsqueeze(targets, 1)
            neg_features = torch.where(mask.any(dim=1, keepdim=True), new_neg_features, neg_features)
            neg_targets = torch.where(mask.any(dim=1), new_neg_targets, neg_targets)
        
            num_tries += 1
    
        return neg_features
        

class Prototypes:
    def __init__(self, n_classes, feature_dim):
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        
        # Initialize prototypes and covariance matrices
        self.prototypes = torch.zeros(n_classes, feature_dim).cuda()
        self.covariances = torch.zeros(n_classes, feature_dim, feature_dim).cuda()
        self.counts = torch.zeros(n_classes).cuda()
        
    def update(self, features, targets):
        # Update prototypes
        fratures=features.cuda()
        targets=targets.cuda()
        for i in range(self.n_classes):
            indices = torch.where(targets == i)[0]
            if len(indices) > 0:
                self.prototypes[i] = F.normalize(features[indices].mean(0), dim=-1)
                self.covariances[i] = torch.matmul((features[indices] - self.prototypes[i]).T, (features[indices] - self.prototypes[i])) / len(indices)
                self.counts[i] += len(indices)
        
        self.prototypes = F.normalize(self.prototypes, dim=-1)
        self.covariances /= self.counts.unsqueeze(-1).unsqueeze(-1)
        
    def get_prototypes(self):
        return self.prototypes
    
    def get_covariances(self):
        return self.covariances