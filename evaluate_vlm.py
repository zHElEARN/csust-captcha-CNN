import argparse
import time
from pathlib import Path
from collections import deque
import tempfile
import torch
from captcha_handler import CaptchaHandler
from vlm_captcha_processor import VLMCaptchaProcessor


class VLMEvaluator:
    def __init__(self, model_path: str, max_requests=100, window_size=10):
        self.handler = CaptchaHandler()
        self.model = VLMCaptchaProcessor(model_path)
        self.max_requests = max_requests

        # 统计相关
        self.total = 0
        self.correct = 0
        self.recent_results = deque(maxlen=window_size)
        self.start_time = None

        # 设备信息
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on: {self.device}")

    def _process_image(self, image_data: bytes) -> tuple:
        """处理验证码图片并返回预测结果"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(image_data)
                image_path = Path(f.name)

            raw_output, cleaned = self.model.process_image(image_path)
            return cleaned
        except Exception as e:
            print(f"图像处理失败: {str(e)}")
            return None
        finally:
            image_path.unlink(missing_ok=True)  # 确保删除临时文件

    def evaluate_single(self):
        """执行单次验证码评估"""
        try:
            image_data = self.handler.get_captcha_image()
            prediction = self._process_image(image_data)

            if not prediction:
                return None, None

            is_correct = self.handler.verify_captcha(prediction)

            # 更新统计
            self.total += 1
            self.correct += int(is_correct)
            self.recent_results.append(is_correct)

            return is_correct, prediction

        except Exception as e:
            print(f"请求失败: {str(e)}")
            return None, None

    def run_evaluation(self):
        """运行评估并显示统计信息"""
        self.start_time = time.time()
        print(f"开始评估（最大请求次数: {self.max_requests}）")
        print("Ctrl+C 终止评估")

        try:
            while self.total < self.max_requests:
                success, prediction = self.evaluate_single()

                if success is not None:
                    status = "✓" if success else "✗"
                    recent_acc = (
                        sum(self.recent_results) / len(self.recent_results)
                        if self.recent_results
                        else 0
                    )
                    print(
                        f"[{self.total}/{self.max_requests}] {status} | "
                        f"预测: {prediction.ljust(8)} | "
                        f"实时准确率: {recent_acc:.1%}"
                    )

        except KeyboardInterrupt:
            print("\n用户终止评估")

        finally:
            duration = time.time() - self.start_time
            print("\n" + "=" * 40)
            print("最终评估结果：")
            print(f"总请求次数: {self.total}")
            print(f"成功次数: {self.correct}")
            if self.total > 0:
                print(f"整体准确率: {self.correct/self.total:.2%}")
                print(f"耗时: {duration:.2f}秒")
                print(f"平均频率: {self.total/duration:.2f}次/秒")
            else:
                print("无有效请求")
            print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="验证码评估工具")
    parser.add_argument("--model-path", required=True, type=str, help="模型路径")
    parser.add_argument("--max-requests", type=int, default=100, help="最大请求次数")
    args = parser.parse_args()

    evaluator = VLMEvaluator(model_path=args.model_path, max_requests=args.max_requests)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
