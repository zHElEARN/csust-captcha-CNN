import torch
from torch.utils.data import DataLoader
from model_config import Config, idx2char
from model import CaptchaModel
from model_data_loader import CaptchaDataset, test_transform


def decode_predictions(outputs):
    preds = torch.stack([o.argmax(1) for o in outputs], 1)
    return ["".join([idx2char[i] for i in sample]) for sample in preds.cpu().numpy()]


def evaluate():
    model = CaptchaModel().to(Config.device)
    model.load_state_dict(
        torch.load(Config.final_model_path, map_location=Config.device)
    )
    model.eval()

    test_set = CaptchaDataset(
        Config.test_captchas_path,
        Config.test_labels_path,
        test_transform,
        return_filename=True,
    )
    test_loader = DataLoader(test_set, Config.batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        for imgs, labels, fnames in test_loader:
            outputs = model(imgs.to(Config.device))
            preds = decode_predictions(outputs)
            truths = [
                "".join([idx2char[i] for i in sample])
                for sample in labels.cpu().numpy()
            ]

            for fname, pred, truth in zip(fnames, preds, truths):
                correct = pred == truth
                results.append(
                    {
                        "file": fname,
                        "prediction": pred,
                        "truth": truth,
                        "correct": correct,
                    }
                )

    # 统计并输出结果
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"Test Accuracy: {accuracy:.4f}")
    return results


if __name__ == "__main__":
    evaluate()
