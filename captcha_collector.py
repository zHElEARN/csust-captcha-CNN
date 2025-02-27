import json
import logging
import tempfile
import uuid
from pathlib import Path
from captcha_handler import CaptchaHandler
from vlm_captcha_processor import VLMCaptchaProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CaptchaCollector:
    """验证码爬取和识别程序"""

    def __init__(
        self,
        model_path: str,
        dataset_name: str,
        target_count: int = 100,
    ):
        self.handler = CaptchaHandler()
        self.model = VLMCaptchaProcessor(model_path)
        self.target_count = target_count
        self.dataset_name = dataset_name
        self.current_count = 0

        # 初始化路径
        self.dataset_dir = Path("datasets")
        self.captchas_dir = self.dataset_dir / f"{dataset_name}_captchas"
        self.labels_path = self.dataset_dir / f"{dataset_name}_labels.json"

        # 创建目录
        self.captchas_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # 加载已有标签
        self.labels = {}
        if self.labels_path.exists():
            try:
                with open(self.labels_path, "r") as f:
                    self.labels = json.load(f)
                logger.info(f"已加载 {len(self.labels)} 条已有记录")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"加载标签文件失败: {str(e)}，将创建新文件")

    def collect_captcha(self) -> None:
        """爬取并识别验证码"""
        while self.current_count < self.target_count:
            try:
                # 生成唯一文件名
                file_name = f"{uuid.uuid4()}.png"
                if file_name in self.labels:
                    continue

                # 获取验证码
                logger.info(f"处理验证码 {file_name}...")
                image_data = self.handler.get_captcha_image()

                # 保存验证码图片
                image_path = self.captchas_dir / file_name
                with open(image_path, "wb") as f:
                    f.write(image_data)

                # 识别验证码
                with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                    temp_file.write(image_data)
                    temp_file.seek(0)
                    raw_output, cleaned = self.model.process_image(Path(temp_file.name))

                if not cleaned:
                    logger.warning(f"验证码 {file_name} 识别结果无效，已跳过")
                    image_path.unlink()  # 删除无效图片
                    continue

                # 更新标签数据
                self.labels[file_name] = cleaned
                self.current_count += 1

                # 实时保存标签
                self._save_labels()
                logger.info(
                    f"成功保存 {file_name} ({self.current_count}/{self.target_count})"
                )

            except Exception as e:
                logger.error(f"处理验证码失败: {str(e)}")
                if "image_path" in locals() and image_path.exists():
                    image_path.unlink()  # 清理失败记录
                continue

    def _save_labels(self) -> None:
        """保存标签文件"""
        try:
            with open(self.labels_path, "w") as f:
                json.dump(self.labels, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存标签文件失败: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="验证码爬取和识别程序")
    parser.add_argument("--model_path", type=str, required=True, help="大模型路径")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument(
        "--target_count", type=int, default=100, help="本次需要追加的验证码数量"
    )

    args = parser.parse_args()

    collector = CaptchaCollector(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        target_count=args.target_count,
    )

    try:
        collector.collect_captcha()
        logger.info(f"成功完成！本次新增 {collector.current_count} 条验证码")
        logger.info(f"数据集总数量: {len(collector.labels)}")
    except KeyboardInterrupt:
        logger.info(f"用户中断！本次已保存 {collector.current_count} 条验证码")
        logger.info(f"数据集总数量: {len(collector.labels)}")
