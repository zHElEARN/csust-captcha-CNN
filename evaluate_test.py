import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from train import CaptchaModel


# 复用训练代码中的配置和模型定义
class Config:
    test_captchas_path = "datasets/test_captchas"
    test_labels_path = "datasets/test_labels.json"
    final_model_path = "final_model.pth"
    num_classes = 36
    max_length = 4
    batch_size = 32  # 可以根据GPU显存调整


class CaptchaDataset(Dataset):
    def __init__(self, captchas_path, labels_path, transform=None):
        with open(labels_path) as f:
            self.labels = json.load(f)
        self.image_files = list(self.labels.keys())
        self.captchas_path = captchas_path
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.captchas_path, img_name))
        label = self.labels[img_name]

        if self.transform:
            image = self.transform(image)

        target = []
        for char in label[: Config.max_length]:
            target.append(char2idx[char.upper()])
        return image, torch.LongTensor(target), img_name  # 增加返回文件名


# 复用训练代码中的字符映射
char2idx = {
    char: idx for idx, char in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
}
idx2char = {idx: char for char, idx in char2idx.items()}

# 测试用数据转换
test_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)


def decode_predictions(outputs):
    """将模型输出解码为字符串"""
    preds = torch.stack([output.argmax(dim=1) for output in outputs], dim=1)
    captcha = []
    for i in range(preds.size(0)):
        chars = [idx2char[idx.item()] for idx in preds[i]]
        captcha.append("".join(chars))
    return captcha


def evaluate_model():
    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = CaptchaModel().to(device)
    model.load_state_dict(torch.load(Config.final_model_path, map_location=device))
    model.eval()

    # 加载测试数据
    test_dataset = CaptchaDataset(
        Config.test_captchas_path, Config.test_labels_path, transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    total = 0
    correct = 0
    results = []

    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = decode_predictions(outputs)

            # 转换真实标签
            true_labels = [
                "".join([idx2char[idx.item()] for idx in sample])
                for sample in labels.cpu()
            ]

            # 统计正确率
            batch_size = images.size(0)
            total += batch_size

            for i in range(batch_size):
                is_correct = preds[i] == true_labels[i]
                correct += int(is_correct)

                # 记录每个样本的结果
                results.append(
                    {
                        "filename": filenames[i],
                        "prediction": preds[i],
                        "true_label": true_labels[i],
                        "correct": is_correct,
                    }
                )

    # 打印详细结果
    print("\nDetailed Results:")
    for result in results:
        status = "✓" if result["correct"] else "✗"
        print(
            f"{status} {result['filename']} - Pred: {result['prediction']} | True: {result['true_label']}"
        )

    # 统计信息
    accuracy = correct / total
    print(f"\nFinal Accuracy on Test Set: {accuracy:.4f} ({correct}/{total})")

    return accuracy


if __name__ == "__main__":
    evaluate_model()
