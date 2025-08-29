import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.lambda_val), None

class DANN(nn.Module):
    def __init__(self, in_channels=1):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 5), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100), nn.ReLU(),
            nn.Linear(100, 10)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100), nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x, lambda_val=1.0):
        features = self.feature_extractor(x)
        features_flat = features.view(-1, 48 * 4 * 4)
        
        # 标签预测
        label_output = self.label_classifier(features_flat)
        
        # 域预测
        reverse_features = GradientReversalLayer.apply(features_flat, lambda_val)
        domain_output = self.domain_classifier(reverse_features)
        
        return label_output, domain_output.squeeze()

# 用于对比的基线模型
class BaselineCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(BaselineCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100), nn.ReLU(), nn.Linear(100, 10)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        features_flat = features.view(-1, 48 * 4 * 4)
        return self.classifier(features_flat)