import torch
import torch.nn as nn
import torchvision.models as models

class TeacherModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(TeacherModel, self).__init__()
        self.encoder = models.resnet50(pretrained=pretrained)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128)
        )

    def forward(self, x):
        features = self.encoder(x)
        # Classification head
        logits = self.classifier(features)
        # Projection head
        projected_features = self.projection(features)
        return features,logits,projected_features


class StudentModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(StudentModel, self).__init__()
        self.encoder = models.resnet50(pretrained=pretrained)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128)
        )


    def forward(self, x):
        features = self.encoder(x)
        # Classification head
        logits = self.classifier(features)
        # Projection head
        projected_features = self.projection(features)
        return features, logits,projected_features