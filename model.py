import torch
import torch.nn as nn

class HybridDehazeNet(nn.Module):
    def __init__(self):
        super(HybridDehazeNet, self).__init__()

        # 특징 추출 (Feature Extraction)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # Conv Layer
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # 지도 정보 인코딩 (Guided Information Encoding)
        self.guided_encoding = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3),  # Larger Kernel for Global Features
            nn.ReLU(inplace=True)
        )

        # 복원 네트워크 (Recovery Network)
        self.recovery = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2)  # Output to RGB
        )

    def forward(self, x):
        x = self.feature_extraction(x)  # 특징 추출
        x = self.guided_encoding(x)    # 지도 정보 인코딩
        x = self.recovery(x)           # 복원 네트워크
        return x
