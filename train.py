import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch import nn

from model_config import Config
from model import CaptchaModel
from model_data_loader import CaptchaDataset, train_transform
from train_utils import train_model, plot_training_history


def main():
    # 初始化数据集
    vlm_dataset = CaptchaDataset(
        Config.vlm_captchas_path, Config.vlm_labels_path, train_transform
    )
    human_dataset = CaptchaDataset(
        Config.human_captchas_path, Config.human_labels_path, train_transform
    )

    # 合并数据集
    full_dataset = ConcatDataset([vlm_dataset, human_dataset])
    train_size = int(0.9 * len(full_dataset))
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, len(full_dataset) - train_size]
    )

    dataloaders = {
        "train": DataLoader(train_dataset, Config.batch_size, shuffle=True),
        "val": DataLoader(val_dataset, Config.batch_size),
    }

    # 初始化模型
    model = CaptchaModel().to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    # 预训练阶段
    print("Starting pretraining...")
    vlm_train_size = int(0.9 * len(vlm_dataset))
    vlm_train, vlm_val = random_split(
        vlm_dataset, [vlm_train_size, len(vlm_dataset) - vlm_train_size]
    )

    pretrain_loaders = {
        "train": DataLoader(vlm_train, Config.batch_size, shuffle=True),
        "val": DataLoader(vlm_val, Config.batch_size),
    }

    pretrain_hist = train_model(
        model,
        pretrain_loaders,
        criterion,
        optimizer,
        Config.num_epochs_pretrain,
        "pretrain",
    )
    torch.save(model.state_dict(), Config.pretrain_model_path)

    # 微调阶段
    print("\nStarting finetuning...")
    for param_group in optimizer.param_groups:
        param_group["lr"] = Config.lr / 10

    # 重新初始化分类层
    for classifier in model.classifiers:
        if isinstance(classifier[-1], nn.Linear):
            nn.init.xavier_normal_(classifier[-1].weight)

    finetune_hist = train_model(
        model, dataloaders, criterion, optimizer, Config.num_epochs_finetune, "finetune"
    )

    # 绘制训练曲线
    plot_training_history(pretrain_hist, finetune_hist)


if __name__ == "__main__":
    main()
