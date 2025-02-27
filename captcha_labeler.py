import json
import os
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk, Image


class CaptchaLabeler:
    def __init__(self, master, dataset_name):
        self.master = master
        self.dataset_name = dataset_name
        master.title(f"验证码标注工具 - {dataset_name}")
        master.geometry("800x600")

        # 初始化路径
        self.labels_path = f"datasets/{dataset_name}_labels.json"
        self.captchas_dir = f"datasets/{dataset_name}_captchas/"

        # 初始化数据
        self.data = self.load_or_init_labels()
        self.keys = list(self.data.keys())
        self.current_index = 0

        # 配置字体
        self.font_style = ("Consolas", 36)
        self.info_font = ("Segoe UI", 12)

        # 创建主界面布局
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # 图片显示区域
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.pack(pady=20)

        # 信息显示
        self.info_label = ttk.Label(self.main_frame, font=self.info_font)
        self.info_label.pack()

        # 输入框
        self.input_var = tk.StringVar()
        self.entry = ttk.Entry(
            self.main_frame,
            textvariable=self.input_var,
            font=self.font_style,
            width=8,
            justify="center",
        )
        self.entry.pack(pady=30)
        self.entry.bind("<Return>", lambda e: self.save_and_move(1))
        self.entry.focus()

        # 控制按钮面板
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(pady=10)
        self.prev_btn = ttk.Button(
            self.control_frame,
            text="上一张 (↑)",
            command=lambda: self.save_and_move(-1),
        )
        self.prev_btn.pack(side="left", padx=10)

        # 绑定快捷键
        master.bind("<Up>", lambda e: self.save_and_move(-1))

        # 加载初始图片
        self.load_image()
        self.update_info()

    def load_or_init_labels(self):
        """加载或初始化标签文件"""
        try:
            if not os.path.exists(self.labels_path):
                return {}
            with open(self.labels_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            messagebox.showerror("错误", f"加载标签文件失败: {str(e)}")
            return {}

    def load_image(self):
        img_name = self.keys[self.current_index]
        img_path = os.path.join(self.captchas_dir, img_name)
        if not os.path.exists(img_path):
            messagebox.showerror("错误", f"图片 {img_path} 不存在！")
            return
        try:
            img = Image.open(img_path)
            img = img.resize((600, 186), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")

        # 设置输入框
        current_value = self.data.get(img_name, "")
        self.input_var.set(current_value)
        self.entry.select_range(0, "end")
        self.entry.focus()

    def update_info(self):
        total = len(self.keys)
        self.info_label.config(
            text=f"数据集: {self.dataset_name}\n"
            f"当前进度: {self.current_index + 1}/{total}\n"
            f"图片名: {self.keys[self.current_index]}"
        )
        self.prev_btn.state(["!disabled" if self.current_index > 0 else "disabled"])

    def save_current(self):
        cleaned = self.input_var.get().strip().upper()
        img_name = self.keys[self.current_index]
        self.data[img_name] = cleaned
        self.input_var.set(cleaned)

    def save_and_move(self, step):
        self.save_current()
        new_index = self.current_index + step
        if 0 <= new_index < len(self.keys):
            self.current_index = new_index
            self.load_image()
            self.update_info()
            self.save_to_json()
        else:
            if step > 0:
                messagebox.showinfo("完成", "所有验证码已处理完成！")
                self.save_to_json()
                self.master.destroy()
            else:
                messagebox.showinfo("提示", "已经是第一张")

    def save_to_json(self):
        try:
            os.makedirs(os.path.dirname(self.labels_path), exist_ok=True)
            with open(self.labels_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证码标注工具")
    parser.add_argument("--name", required=True, help="数据集名称（不含后缀）")
    args = parser.parse_args()

    # 验证数据集是否存在
    captchas_dir = f"datasets/{args.name}_captchas/"
    if not os.path.exists(captchas_dir):
        messagebox.showerror("错误", f"数据集目录 {captchas_dir} 不存在！")
        exit(1)

    root = tk.Tk()
    style = ttk.Style()
    style.configure("TButton", font=("Segoe UI", 12))
    style.configure("TEntry", font=("Consolas", 36))
    app = CaptchaLabeler(root, args.name)
    root.mainloop()
