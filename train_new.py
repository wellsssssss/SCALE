import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from data import source_loader,target_loader,source_dataset,target_dataset
from net import TeacherModel,StudentModel
from tqdm import tqdm
import matplotlib as plt
from torch.optim.lr_scheduler import StepLR
from proto import MemoryBank,Prototypes
from loss import ContrastiveLoss,RegularizationLoss,compute_ssl_loss
from protocl_loss import Protocl,compute_prototypes
from lreg import LregLoss
from distcl_loss import dist_contrastive_cls
device = torch.device('cuda')
def cross_entropy_loss(logits, targets):
    return F.cross_entropy(logits, targets)

def main():
    num_epochs = 50
    feature_dim=128
    ema_decay = 0.99
    learning_rate = 2e-5
    warmup_iterations = 5000
    memory_bank_size = 200
    num_classes=65
    lambda_cl = 0.1
    lambda_reg = 0.1
    ssl_loss= torch.tensor(0.0).cuda()
    prototypes = torch.zeros(65, 128).cuda()
    cl_loss = torch.tensor(0.0).cuda()
    reg_loss = torch.tensor(0.0).cuda()
    protocl_loss=torch.tensor(0.0).cuda()
    dist_loss=torch.tensor(0.0).cuda()
    total_losses = []
    target_accuracies = []
    iteration=1
    lreg_loss = LregLoss(num_classes)
    # 初始化教师网络和学生网络
    teacher_model = TeacherModel(num_classes=len(source_dataset.classes)).cuda()
    student_model = StudentModel(num_classes=len(source_dataset.classes)).cuda()
    # protocl = Protocl(student_model).cuda()
    encoder_params = list(student_model.encoder.parameters())
    classifier_params = list(student_model.classifier.parameters())
    # 定义优化器
    optimizer = optim.Adam(encoder_params + classifier_params, lr=learning_rate, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
#     # 定义ContrastiveLoss和RegularizationLoss
#     contrast_criterion = ContrastiveLoss(temperature=0.07)
#     regularization_criterion = RegularizationLoss(temperature=0.07)
    # dist=Distcl(num_classes,feature_dim,student_model,source_loader,device)
    for epoch in range(num_epochs):
        teacher_model.eval()
        student_model.train()
        epoch_loss = 0.0
        mean_dict = {i: torch.zeros_like(torch.tensor(feature_dim)) for i in range(num_classes)}
        covariance_dict = {i: torch.zeros_like(torch.tensor(feature_dim)) for i in range(num_classes)}
        # if iteration > warmup_iterations:
        #     prototypes = compute_prototypes(student_model, source_loader)
        progress_bar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)), desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, ((source_images, source_labels), (target_images, _)) in enumerate(progress_bar):
            # iteration = epoch * len(progress_bar) + batch_idx
            if iteration >= warmup_iterations:
                optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-4)
            source_images, source_labels = source_images.cuda(), source_labels.cuda()
            target_images = target_images.cuda()
            with torch.no_grad():
                _, teacher_logits, _ = teacher_model(target_images)
                pseudo_labels = torch.argmax(teacher_logits, dim=1)
            source_features, source_logits, source_projected = student_model(source_images)
            target_features, target_logits, target_projected = student_model(target_images)
            for i in range(num_classes):
                mask = source_labels == i
                if mask.sum() > 0:
                    class_features = source_projected[mask]
                    class_mean = class_features.mean(dim=0)
                    class_covariance = (class_features - class_mean).t().matmul(class_features - class_mean) / (class_features.size(0) - 1)
                
                    if batch_idx == 0:
                        mean_dict[i] = class_mean
                        covariance_dict[i] = class_covariance
                    else:
                        mean_dict[i] = (mean_dict[i] * batch_idx + class_mean) / (batch_idx + 1)
                        covariance_dict[i] = (covariance_dict[i] * batch_idx + class_covariance) / (batch_idx + 1)
            ce_loss = cross_entropy_loss(source_logits, source_labels)
            if iteration > warmup_iterations:
                # prototypes = compute_prototypes(student_model, source_loader)
                # protocl_loss = protocl(target_images, pseudo_labels,prototypes)
                dist_loss=dist_contrastive_cls(target_projected, pseudo_labels, mean, covariance)
            # ssl_loss = torch.tensor(0.0).cuda()
            # if iteration > warmup_iterations:
            
            # ssl_loss = compute_ssl_loss(teacher_logits, target_logits, threshold=0.8)
                
            # with torch.no_grad():
            #     memory_bank.update(source_projected.detach(), source_labels)
            #     prototypes.update(source_projected.detach(), source_labels)
                
            # cl_loss = torch.tensor(0.0).cuda()
            # if iteration > warmup_iterations:
            #     source_neg = memory_bank.sample(source_images.shape[0], source_labels)
            #     target_neg = memory_bank.sample(target_images.shape[0], pseudo_labels)
            #     source_projected = source_projected.unsqueeze(1)
            #     target_projected = target_projected.unsqueeze(1)
            #     cl_loss = contrast_criterion(torch.cat([source_projected, target_projected], dim=0),
            #                                  torch.cat([source_labels, pseudo_labels], dim=0),
            #                                  prototypes=prototypes.get_prototypes(),
            #                                  covariances=prototypes.get_covariances())
            # reg_loss = torch.tensor(0.0).cuda()
            # if iteration > warmup_iterations:
            #     reg_loss = regularization_criterion(target_projected,prototypes.get_prototypes())
                
            # total_loss = ce_loss+ssl_loss+lambda_cl*cl_loss+lambda_reg*reg_loss
            total_loss = ce_loss+ssl_loss+0.1*protocl_loss+0.1*dist_loss
            total_loss.backward()
            all_parameters = list(student_model.parameters())  # 假设你要更新学生模型的参数
            gradients = [param.grad for param in all_parameters if param.grad is not None]
            max_norm = max([torch.norm(g.detach()) for g in gradients])
            normalized_gradients = [g / torch.norm(g.detach()) * max_norm for g in gradients]

        # 手动将规范化的梯度赋给模型参数
            for param, grad in zip(all_parameters, normalized_gradients):
                param.grad = grad

        # 使用优化器来更新模型参数
            optimizer.step()
            optimizer.zero_grad()
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.data = ema_decay * teacher_param.data + (1 - ema_decay) * student_param.data

            epoch_loss += total_loss.item()
            progress_bar.set_postfix({"Loss": total_loss.item(),
                                      "CE Loss": ce_loss.item(),
                                      "SSL Loss": ssl_loss.item(),
                                      "CL Loss": protocl_loss.item(),
                                      "Reg Loss": reg_loss.item()})
            total_losses.append(epoch_loss / (batch_idx + 1))
        mean = torch.stack([mean_dict[i] for i in range(num_classes)]).cuda() 
        covariance = torch.stack([covariance_dict[i] for i in range(num_classes)]).cuda()
                # 在源域上评估学生网络的性能
        student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in target_loader:
                images, labels = images.cuda(), labels.cuda()
                _, logits,_ = student_model(images)
                predictions = torch.argmax(logits, dim=1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = 100 * correct / total
        target_accuracies.append(accuracy)
        if accuracy>100:
            iteration=150090
        print(f"Target Domain Accuracy: {accuracy:.2f}%")
        
if __name__ == '__main__':
    main()