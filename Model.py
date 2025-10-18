"""
    This file contains the model for the application.
    The model contains:
     1. A simple NN classification model.
     2. A simple CNN classification model.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """
    简单的全连接神经网络分类模型
    包含2个全连接层

    输入尺寸: batch, 3, 156, 128 (通道, 高, 宽)
    """

    def __init__(self, input_size, hidden_size, num_classes=40):
        """
        初始化网络结构

        参数:
            input_size: 输入特征维度 (3*156*128)
            hidden_size: 隐藏层神经元数量
            num_classes: 输出类别数(默认40类)
        """
        super(SimpleNN, self).__init__()
        # 第一全连接层: 输入维度 -> 隐藏层维度
        self.fc1 = nn.Linear(input_size, hidden_size)  # 公式: Θ^T X + b
        # 第二全连接层: 隐藏层维度 -> 类别数
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        前向传播过程

        参数:
            x: 输入张量 [batch, 3, 156, 128]
        返回:
            输出张量 [batch, num_classes]
        """
        # 展平操作: [batch, 3, 156, 128] -> [batch, 3*156*128]
        x = x.view(x.size(0), -1)
        # 第一层全连接 + ReLU激活
        out = self.fc1(x)
        out = F.relu(out)
        # 第二层全连接 (无激活函数，输出原始logits)
        out = self.fc2(out)
        return out


class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络分类模型
    包含2个卷积层和1个全连接层

    输入尺寸: 3, 156, 128 (通道, 高, 宽)
    全连接层输入尺寸: 32*39*32 (自动计算)
    """

    def __init__(self, num_classes=40):
        """
        初始化网络结构

        参数:
            num_classes: 输出类别数(默认40类)
        """
        super(SimpleCNN, self).__init__()
        # 第一卷积块: Conv2d -> ReLU -> MaxPool
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # 保持空间维度
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 下采样到原尺寸1/2

        # 第二卷积块: Conv2d -> ReLU -> MaxPool
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # 通道数16->32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 再次下采样到1/4

        # 全连接层: 32*39*32 -> num_classes
        # 计算说明:
        # 原始输入: 156x128
        # 第一次池化后: 78x64
        # 第二次池化后: 39x32
        # 通道数: 32
        self.fc = nn.Linear(32 * 39 * 32, num_classes)

    def forward(self, x):
        """
        前向传播过程

        参数:
            x: 输入张量 [batch, 3, 156, 128]
        返回:
            输出张量 [batch, num_classes]
        """
        # 第一卷积块
        out = self.layer1(x)
        # 第二卷积块
        out = self.layer2(out)
        # 展平操作: [batch, 32, 39, 32] -> [batch, 32*39*32]
        out = out.reshape(out.size(0), -1)
        # 全连接层
        out = self.fc(out)
        return out