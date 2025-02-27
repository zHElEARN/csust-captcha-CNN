import torch
from PIL import Image
import io
import time
from collections import deque
from torchvision import transforms

from train import CaptchaModel, Config
from captcha_handler import CaptchaHandler

char2idx = {
    char: idx for idx, char in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
}
idx2char = {idx: char for char, idx in char2idx.items()}


class OnlineEvaluator:
    def __init__(self, max_requests=100, window_size=10):
        self.handler = CaptchaHandler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )

        # 统计相关
        self.total = 0
        self.correct = 0
        self.recent_results = deque(maxlen=window_size)
        self.max_requests = max_requests

    def _load_model(self):
        """加载训练好的模型"""
        model = CaptchaModel().to(self.device)
        model.load_state_dict(
            torch.load(Config.final_model_path, map_location=self.device)
        )
        model.eval()
        return model

    def _preprocess_image(self, image_data: bytes):
        """预处理从接口获取的图片"""
        image = Image.open(io.BytesIO(image_data))
        return self.transform(image).unsqueeze(0).to(self.device)  # 添加batch维度

    def _predict(self, image_tensor):
        """执行预测"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            preds = torch.stack([output.argmax(dim=1) for output in outputs], dim=1)
            return "".join([idx2char[idx.item()] for idx in preds[0]])

    def evaluate_single(self):
        """执行单次验证码评估"""
        try:
            # 获取验证码
            image_data = self.handler.get_captcha_image()

            # 预处理并预测
            image_tensor = self._preprocess_image(image_data)
            prediction = self._predict(image_tensor)

            # 验证结果
            is_correct = self.handler.verify_captcha(prediction)

            # 更新统计
            self.total += 1
            self.correct += int(is_correct)
            self.recent_results.append(is_correct)

            return is_correct, prediction

        except Exception as e:
            print(f"请求失败: {str(e)}")
            return None, None

    def run_continuous_evaluation(self):
        """持续运行评估并显示统计信息"""
        start_time = time.time()

        print(f"开始在线评估（最大请求次数: {self.max_requests}）")
        print("Ctrl+C 终止评估")

        try:
            while self.total < self.max_requests:
                success, prediction = self.evaluate_single()

                # 实时显示统计
                if success is not None:
                    status = "✓" if success else "✗"
                    print(
                        f"[{self.total}/{self.max_requests}] {status} | "
                        f"预测结果: {prediction}"
                    )

        except KeyboardInterrupt:
            print("\n用户终止评估")

        finally:
            duration = time.time() - start_time
            print("\n" + "=" * 40)
            print("最终评估结果：")
            print(f"总请求次数: {self.total}")
            print(f"成功次数: {self.correct}")
            print(
                f"整体准确率: {self.correct/self.total:.2%}"
                if self.total > 0
                else "无有效请求"
            )
            print(f"耗时: {duration:.2f}秒")
            print(f"平均频率: {self.total/duration:.2f}次/秒" if duration > 0 else "")
            print("=" * 40)


if __name__ == "__main__":
    evaluator = OnlineEvaluator(max_requests=500)
    evaluator.run_continuous_evaluation()
