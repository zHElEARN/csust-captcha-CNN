import re
import json
import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class CaptchaHandler:
    CAPTCHA_URL = "https://login.csust.edu.cn:802/eportal/captcha"
    VERIFY_URL = "https://login.csust.edu.cn:802/eportal/portal/captcha/check"

    def __init__(self) -> None:
        self.session = requests.Session()

    def get_captcha_image(self) -> bytes:
        """获取验证码图片二进制数据"""
        try:
            response = self.session.get(
                self.CAPTCHA_URL,
                stream=True,
                timeout=10,
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.error(f"获取验证码失败: {e}")
            raise RuntimeError("无法获取验证码")

    def verify_captcha(self, captcha: str) -> bool:
        """验证码校验"""
        params = {
            "callback": "dr1003",
            "captcha": captcha,
            "jsVersion": "4.2.1",
            "v": "5470",
            "lang": "zh",
        }
        try:
            response = self.session.get(
                self.VERIFY_URL,
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = self._parse_callback(response.text)
        except (requests.RequestException, ValueError) as e:
            logger.error(f"验证码校验失败: {e}")
            return False
        return data.get("result") == 1

    @staticmethod
    def _parse_callback(text: str) -> Dict[str, Any]:
        """解析回调函数响应"""
        match = re.match(r"^\w+\((.*)\);$", text.strip())
        if not match:
            raise ValueError("无效的响应格式")
        return json.loads(match.group(1))


# 使用示例
if __name__ == "__main__":
    handler = CaptchaHandler()

    try:
        # 获取验证码图片
        image_data: bytes = handler.get_captcha_image()

        # 处理验证码图片
        captcha: str = "1234"

        # 验证验证码
        if handler.verify_captcha(captcha):
            print("验证码校验成功！")
        else:
            print("验证码校验失败")

    except Exception as e:
        print(f"处理失败: {str(e)}")
