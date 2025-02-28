import torch
import matplotlib.pyplot as plt
from model_config import Config


def train_model(model, dataloaders, criterion, optimizer, num_epochs, phase="pretrain"):
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    model = model.to(Config.device)

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
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = sum(
                        criterion(o, labels[:, i]) for i, o in enumerate(outputs)
                    )

                    preds = torch.stack([o.argmax(1) for o in outputs], dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).all(1).sum().item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            key = f"{phase}_loss"
            history[key].append(epoch_loss)
            history[key.replace("loss", "acc")].append(epoch_acc)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), Config.final_model_path)

        # 预训练早停逻辑
        if phase == "pretrain" and epoch_acc > 0.15 and epoch > 20:
            print(f"Early stopping at epoch {epoch}")
            break

    return history


def plot_training_history(pretrain_hist, finetune_hist):
    plt.figure(figsize=(12, 8))

    # 预训练部分绘图
    plt.subplot(2, 2, 1)
    plt.plot(pretrain_hist["train_loss"], "b", label="Training")
    plt.plot(pretrain_hist["val_loss"], "r", label="Validation")
    plt.title("Pretraining Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(pretrain_hist["train_acc"], "b", label="Training")
    plt.plot(pretrain_hist["val_acc"], "r", label="Validation")
    plt.title("Pretraining Accuracy")
    plt.legend()

    # 微调部分绘图
    plt.subplot(2, 2, 3)
    plt.plot(finetune_hist["train_loss"], "b", label="Training")
    plt.plot(finetune_hist["val_loss"], "r", label="Validation")
    plt.title("Finetuning Loss")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(finetune_hist["train_acc"], "b", label="Training")
    plt.plot(finetune_hist["val_acc"], "r", label="Validation")
    plt.title("Finetuning Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
