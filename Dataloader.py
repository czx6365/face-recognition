"""
    Pytorch dataloader for the celebA dataset

    The dataset is downloaded from the following link:
        https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
BASE_PATH = "F:\人脸识别比赛\CelebA"
FILE_PATH = {
    "eval_partition": f'{BASE_PATH}/Dataset/Eval/list_eval_partition.txt',
    "anno_identity":f"{BASE_PATH}/Dataset/Anno/identity_CelebA.txt",
    "anno_list_attr": f"{BASE_PATH}/Dataset/Anno/list_attr_celeba.txt",
    "anno_bbox": f"{BASE_PATH}/Dataset/Anno/list_bbox_celeba.txt",
    "anno_landmarks": f"{BASE_PATH}/Dataset/Anno/list_landmarks_align_celeba.txt", # This file is not used
    "anno_landmarks_untouched": f"{BASE_PATH}/Dataset/Anno/list_landmarks_celeba.txt", # This file is not used
    "img_dir": f"{BASE_PATH}/Dataset/Img/img_align_celeba/",
}

DefaultTransform = transforms.Compose([
    transforms.Resize(128),
    # transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # 0-255 -> 0-1
])

class CelebADataset(Dataset):
    """CelebA dataset."""

    def __getitem__(self, idx):
        """
        根据索引获取数据样本的方法
        参数:idx: 样本索引
        返回: image: 处理后的图像张量
            label: 处理后的标签张量
        """
        # 1. 获取图像路径
        # 拼接图像完整路径：基础路径 + 当前数据中第idx行第0列的文件名
        img_name = os.path.join(self.file_path['img_dir'], self.current_data.iloc[idx, 0])
        # 2. 加载图像
        image = Image.open(img_name)
        # 3. 加载属性标签
        # 获取当前数据中第idx行从第2列开始的所有列作为标签（假设前两列是文件名和其他信息）
        label = self.current_data.iloc[idx, 2:].values.astype('float')
        # 将标签转换为float32类型的PyTorch张量
        label = torch.tensor(label, dtype=torch.float32)
        # 4. 将标签转换为one-hot编码形式
        # 原始数据中-1表示不适用/负样本，转换为0
        label = torch.where(label == -1, torch.tensor(0), label)
        # 原始数据中1表示正样本，保持为1
        label = torch.where(label == 1, torch.tensor(1), label)
        # 5. 应用图像变换（如果有）
        if self.transform:
            image = self.transform(image)
        # 返回处理后的图像和标签
        return image, label
    
    def __len__(self):
        return len(self.current_data)

    def __init__(self, transform=DefaultTransform, train_mode="train"):
        """
        数据集初始化方法

        参数:
            transform (callable, 可选): 应用于样本的可选数据变换，默认为DefaultTransform
            train_mode (string): 数据集模式，可选"train"(训练)、"val"(验证)或"test"(测试)
        """
        # 1. 初始化文件路径和变换
        self.file_path = FILE_PATH  # 预设的文件路径配置
        self.transform = transform  # 图像预处理变换方法

        # 2. 加载元数据
        self.data = self._load_data()  # 调用内部方法加载并合并所有数据

        # 3. 根据模式设置当前使用的数据子集
        self.current_data = None  # 初始化当前数据为空

        # 根据train_mode选择不同分区的数据:
        # partition列中0=训练集，1=验证集，2=测试集
        if train_mode == "train":
            self.current_data = self.data[self.data['partition'] == 0]  # 筛选训练集
        elif train_mode == "val":
            self.current_data = self.data[self.data['partition'] == 1]  # 筛选验证集
        elif train_mode == "test":
            self.current_data = self.data[self.data['partition'] == 2]  # 筛选测试集
        else:
            raise ValueError("Invalid train mode")  # 非法模式报错

    def _load_data(self):
        """
        内部方法：加载并合并分区文件和属性文件

        返回:
            pd.DataFrame: 合并后的完整数据集
        """
        # 1. 加载数据分区文件
        # 文件格式：每行包含图片ID和分区标识(0/1/2)
        # 参数说明：
        #   delim_whitespace=True - 使用空白符分隔
        #   header=None - 无表头
        #   names - 指定列名
        partition = pd.read_csv(
            self.file_path['eval_partition'],
            delim_whitespace=True,
            header=None,
            names=['img_id', 'partition']
        )
        # 确保数据类型正确
        partition['partition'] = partition['partition'].astype(int)
        partition['img_id'] = partition['img_id'].astype(str)

        # 2. 加载属性标注文件
        # 文件格式：第一行是属性数量，第二行开始是属性名称，之后每行对应一张图片的属性
        attr = pd.read_csv(
            self.file_path['anno_list_attr'],
            delim_whitespace=True,
            header=1  # 第一行作为列名
        )
        # 添加img_id列（使用行索引作为图片ID）
        attr['img_id'] = attr.index.astype(str)

        # 3. 合并两个DataFrame（通过img_id列）
        self.data = pd.merge(partition, attr, on='img_id')
        return self.data

if __name__ == "__main__":
    dataset = CelebADataset()
    print(len(dataset))
    img, label = dataset[0]
    # label: [0, 1, 0, 1, ...]
    # prediction: [0-1, 0-1, 0-1, ...]
    # question: how to build a model to predict the label
    # model: 
    # input: img, output: prediction
    # plt img
    import matplotlib.pyplot as plt
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print(label)
