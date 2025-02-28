import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt


# 配置参数
class Config:
    # 数据集配置
    vlm_captchas_path = "datasets/vlm_captchas"
    vlm_labels_path = "datasets/vlm_labels.json"
    human_captchas_path = "datasets/human_captchas"
    human_labels_path = "datasets/human_labels.json"

    # 训练参数
    batch_size = 64
    lr = 0.001
    num_epochs_pretrain = 50  # 预训练轮数
    num_epochs_finetune = 100  # 微调轮数
    num_classes = 36  # 26字母+10数字
    max_length = 4  # 验证码长度

    # 模型保存路径
    pretrain_model_path = "pretrain_model.pth"
    final_model_path = "final_model.pth"


# 字符映射表
char2idx = {
    char: idx for idx, char in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
}
idx2char = {idx: char for char, idx in char2idx.items()}

# 数据增强
train_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)


# 自定义Dataset
class CaptchaDataset(Dataset):
    def __init__(self, captchas_path, labels_path, transform=None):
        with open(labels_path) as f:
            self.labels = json.load(f)
        self.image_files = list(self.labels.keys())
        self.captchas_path = captchas_path
        self.transform = transform

        # 统计标签长度
        lengths = defaultdict(int)
        for label in self.labels.values():
            lengths[len(label)] += 1
        print(f"Label length distribution: {dict(lengths)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.captchas_path, img_name))
        label = self.labels[img_name]

        if self.transform:
            image = self.transform(image)

        # 将标签转换为四个分类任务的目标
        target = []
        for char in label[: Config.max_length]:
            target.append(char2idx[char.upper()])
        return image, torch.LongTensor(target)


# 模型结构
class CaptchaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 6)),
        )

        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 6, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, Config.num_classes),
                )
                for _ in range(Config.max_length)
            ]
        )

    def forward(self, x):
        features = self.cnn(x)
        return [cls(features) for cls in self.classifiers]


# 训练函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs, phase="pretrain"):
    best_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = 0
                    for i in range(Config.max_length):
                        loss += criterion(outputs[i], labels[:, i])

                    preds = torch.stack(
                        [output.argmax(dim=1) for output in outputs], dim=1
                    )

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.all(preds == labels, dim=1).sum().item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            # 保存最佳模型
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), Config.final_model_path)

        # 预训练阶段提前停止判断
        if phase == "pretrain" and epoch_acc > 0.15 and epoch > 20:
            print(f"Early stopping at epoch {epoch}")
            break

    return (
        model,
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
    )


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1. 加载数据集
    vlm_dataset = CaptchaDataset(
        Config.vlm_captchas_path, Config.vlm_labels_path, transform=train_transform
    )
    human_dataset = CaptchaDataset(
        Config.human_captchas_path, Config.human_labels_path, transform=train_transform
    )

    # 合并数据集并划分
    full_dataset = ConcatDataset([vlm_dataset, human_dataset])
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=Config.batch_size),
    }

    # 2. 初始化模型
    model = CaptchaModel().to(device)

    # 3. 预训练阶段（仅使用VLM数据）
    print("Starting pretraining...")
    vlm_train_size = int(0.9 * len(vlm_dataset))
    vlm_train, vlm_val = torch.utils.data.random_split(
        vlm_dataset, [vlm_train_size, len(vlm_dataset) - vlm_train_size]
    )

    pretrain_dataloaders = {
        "train": DataLoader(vlm_train, batch_size=Config.batch_size, shuffle=True),
        "val": DataLoader(vlm_val, batch_size=Config.batch_size),
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    # 第一阶段预训练
    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = (
        train_model(
            model,
            pretrain_dataloaders,
            criterion,
            optimizer,
            Config.num_epochs_pretrain,
            "pretrain",
        )
    )

    # 保存预训练模型
    torch.save(model.state_dict(), Config.pretrain_model_path)

    # 4. 微调阶段（使用全部数据）
    print("\nStarting fine-tuning...")
    # 降低学习率
    for param_group in optimizer.param_groups:
        param_group["lr"] = Config.lr / 10

    # 重新初始化最后的分类层（可选）
    for classifier in model.classifiers:
        if isinstance(classifier[-1], nn.Linear):
            nn.init.xavier_normal_(classifier[-1].weight)

    model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = (
        train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            Config.num_epochs_finetune,
            "finetune",
        )
    )

    print(f"Training complete. Best model saved to {Config.final_model_path}")

    # 绘制并保存训练和验证的损失和准确率图像
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, "b", label="Training loss")
    plt.plot(epochs, val_loss_history, "r", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, "b", label="Training Accuracy")
    plt.plot(epochs, val_acc_history, "r", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


if __name__ == "__main__":
    main()
