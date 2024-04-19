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
from loss import ContrastiveLoss,RegularizationLoss
device = torch.device('cuda')
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def main():
    num_epochs = 50
    ema_decay = 0.99
    learning_rate = 2e-5
    warmup_iterations = 1000
    memory_bank_size = 1024
    lambda_cl = 1.0
    lambda_reg = 0.1

    total_losses = []
    target_accuracies = []

    teacher_model = TeacherModel(num_classes=len(source_dataset.classes)).cuda()
    student_model = StudentModel(num_classes=len(source_dataset.classes)).cuda()

    # 定义优化器
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # 定义memory bank和prototypes
    memory_bank = MemoryBank(len(source_dataset.classes), feature_dim=128, memory_size=memory_bank_size)
    prototypes = Prototypes(len(source_dataset.classes), feature_dim=128)

    # 定义ContrastiveLoss和RegularizationLoss
    contrast_criterion = ContrastiveLoss(temperature=0.07)
    regularization_criterion = RegularizationLoss(temperature=0.07)

    # 训练循环
    for epoch in range(num_epochs):
        teacher_model.eval()
        student_model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)), desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, ((source_images, source_labels), (target_images, _)) in enumerate(progress_bar):
            iteration = epoch * len(progress_bar) + batch_idx
            source_images, source_labels = source_images.cuda(), source_labels.cuda()
            target_images = target_images.cuda()

            # 教师网络的前向传播
            with torch.no_grad():
                _, teacher_logits, _ = teacher_model(target_images)
                pseudo_labels = torch.argmax(teacher_logits, dim=1)

            # 学生网络的前向传播
            source_features, source_logits, source_projected = student_model(source_images)
            target_features, target_logits, target_projected = student_model(target_images)
            # 计算交叉熵损失
            ce_loss = F.cross_entropy(source_logits, source_labels)
            # 计算自监督损失
            ssl_loss = torch.tensor(0.0).cuda()
            if iteration > warmup_iterations:
                ssl_loss = F.cross_entropy(target_logits, pseudo_labels)

            # 更新memory bank和prototypes
            with torch.no_grad():
                memory_bank.update(source_projected.detach(), source_labels)
                prototypes.update(source_projected.detach(), source_labels)

            # 计算对比损失
            cl_loss = torch.tensor(0.0).cuda()
            if iteration > warmup_iterations:
                source_neg = memory_bank.sample(source_images.shape[0], source_labels)
                target_neg = memory_bank.sample(target_images.shape[0], pseudo_labels)
                source_projected = source_projected.unsqueeze(1)
                target_projected = target_projected.unsqueeze(1)
                cl_loss = contrast_criterion(torch.cat([source_projected, target_projected], dim=0),
                                             torch.cat([source_labels, pseudo_labels], dim=0),
                                             prototypes=prototypes.get_prototypes(),
                                             covariances=prototypes.get_covariances())

            # 计算正则化损失
            # print(source_projected_mean.size())
            # source_projected_mean = source_projected.mean(dim=0, keepdim=True)
            # target_projected_mean = target_projected.mean(dim=0, keepdim=True)
            # projected_mean = torch.cat([source_projected_mean, target_projected_mean], dim=0).mean(0, keepdim=True)
            # print(source_projected_mean.size())
            reg_loss = torch.tensor(0.0).cuda()
            if iteration > warmup_iterations:
                reg_loss = regularization_criterion(target_projected,prototypes.get_prototypes())

            # 总损失
            loss = ce_loss + ssl_loss + lambda_cl * cl_loss + lambda_reg * reg_loss
            loss.backward()
            epoch_loss += loss.item()
            
            all_parameters = list(student_model.parameters())  # 假设你要更新学生模型的参数
            gradients = [param.grad for param in all_parameters if param.grad is not None]
            max_norm = max([torch.norm(g.detach()) for g in gradients])
            normalized_gradients = [g / torch.norm(g.detach()) * max_norm for g in gradients]

        # 手动将规范化的梯度赋给模型参数
            for param, grad in zip(all_parameters, normalized_gradients):
                param.grad = grad
            # 反向传播和优化
            optimizer.step()
            optimizer.zero_grad()


            # EMA更新教师模型
            update_ema_variables(teacher_model, student_model, ema_decay, iteration)

            # 更新进度条
            progress_bar.set_postfix({"Loss": loss.item(),
                                      "CE Loss": ce_loss.item(),
                                      "SSL Loss": ssl_loss.item(),
                                      "CL Loss": cl_loss.item(),
                                      "Reg Loss": reg_loss.item()})

        # 学习率衰减
        scheduler.step()

        # 评估模型
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
        print(f"Target Domain Accuracy: {accuracy:.2f}%")
    # 训练完成后的操作
    # print("Training finished!")
    # print(f"Best Target Accuracy: {max(target_accuracies):.4f}")
    
if __name__ == '__main__':
    main()