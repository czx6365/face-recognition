"""
This file is used to train the model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataloader import CelebADataset, DefaultTransform
from Model import SimpleNN, SimpleCNN
import os
import numpy as np
import pandas as pd
import argparse
import time
import copy
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    模型训练函数
    参数:
        model: 待训练的神经网络模型
        dataloaders: 包含训练集和验证集的DataLoader字典
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数(默认25)
    返回:
        model: 训练完成的模型
        val_acc_history: 验证集准确率历史记录
    """
    since = time.time()  # 记录训练开始时间
    val_acc_history = []  # 存储验证集准确率历史

    # 初始化最佳模型权重和准确率
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 开始训练循环
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch包含训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式(启用dropout等)
            else:
                model.eval()  # 设置为评估模式

            running_loss = []  # 记录当前阶段的损失
            running_accs = []  # 记录当前阶段的准确率

            # 使用进度条遍历数据
            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                # 将数据转移到指定设备(CPU/GPU)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清空梯度缓存
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = F.sigmoid(outputs)  # 使用sigmoid将输出映射到0-1
                    loss = criterion(outputs, labels)

                    # 生成预测结果(>0.5视为正类)
                    preds = torch.where(outputs > 0.5,
                                        torch.tensor(1).to(device),
                                        torch.tensor(0).to(device))

                    # 只在训练阶段执行反向传播和优化
                    if phase == 'train':
                        loss.backward()  # 反向传播计算梯度
                        optimizer.step()  # 更新权重

                # 统计指标
                running_loss.append(loss.item() * inputs.size(0))  # 累计损失
                # 计算准确率(匹配的标签数/(样本数*40个属性))
                running_accs.append((torch.sum(preds == labels.data) /
                                     (len(labels) * 40)).to("cpu").numpy())

            # 计算epoch平均损失和准确率
            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_accs)

            # 打印阶段结果
            print('phase: {} epoch: {} Loss: {:.4f} Acc: {:.4}'.format(
                phase, epoch, epoch_loss, epoch_acc))

            # 如果是验证阶段且当前模型更好，则保存模型权重
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # 记录验证准确率历史
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        # 训练结束后打印耗时
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # 加载最佳模型权重
        model.load_state_dict(best_model_wts)

    return model, val_acc_history

def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--model', type=str, default="SimpleNN", help='Model to train')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    
    # Load the data
    dataloaders = {
        'train': DataLoader(CelebADataset(transform=DefaultTransform, train_mode="train"), batch_size=args.batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(CelebADataset(transform=DefaultTransform, train_mode="val"), batch_size=args.batch_size, shuffle=True, num_workers=4)
    }
    # access the data: dataloaders['train'][0] (load batch of data)
    
    # Load the model
    if args.model == "SimpleNN":
        model = SimpleNN(3*156*128, 300)
    elif args.model == "SimpleCNN":
        model = SimpleCNN()
    elif args.model == "ResNet50":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        model.fc = nn.Linear(2048, 40)
    else:
        raise ValueError("Invalid model")
    
    # Move the model to the device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Train the model
    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=args.num_epochs)
    
    # Save the model
    model_path = os.path.join("models", args.model + ".pth")
    torch.save(model.state_dict(), model_path)
    
    # Plot the accuracy
    plt.plot(hist)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.show()


if __name__ == "__main__":
    main()