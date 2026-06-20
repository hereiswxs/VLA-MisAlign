
import time

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from autoadv_model import PerturbationGenerator
from typing import List, Dict
import shutil
from datetime import datetime


def get_today_str():
    """获取当前日期字符串，格式：YYYY-MM-DD"""
    return datetime.now().strftime('day_%Y_%m_%d')

def ensure_dir(path):
    """如果路径不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


def copy_file_to_dir(src_file, dst_dir):
    """将文件复制到目标目录"""
    ensure_dir(dst_dir)
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_dir)
        print(f"Copied {src_file} to {dst_dir}")
    else:
        print(f"Source file not found: {src_file}")


def count_frames_per_demo(root_dir):
    frame_counts = []

    # 遍历所有任务文件夹
    for task_name in sorted(os.listdir(root_dir)):
        task_path = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        # 遍历每个任务文件夹下的 demo 文件夹
        for demo_name in sorted(os.listdir(task_path)):
            demo_path = os.path.join(task_path, demo_name)
            if not os.path.isdir(demo_path):
                continue

            # 统计该 demo 文件夹中所有 .png 文件
            img_files = [f for f in os.listdir(demo_path) if f.endswith('.png')]
            frame_counts.append({
                "task": task_name,
                "demo": demo_name,
                "frames": len(img_files)
            })

    return frame_counts




DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def save_noised_images(labels, datas, output_root, model):
    noised_root = os.path.join(output_root, 'noised_image')
    noise_only_root = os.path.join(output_root, 'noise_only_image')

    os.makedirs(noised_root, exist_ok=True)
    os.makedirs(noise_only_root, exist_ok=True)

    transform_to_tensor = transforms.ToTensor()
    transform_to_pil = transforms.ToPILImage()

    model.eval()
    with torch.no_grad():
        for label, demo_data in zip(labels, datas):
            task_name, demo_name = label.split('/')
            save_dir_noised = os.path.join(noised_root, task_name, demo_name)
            save_dir_noise_only = os.path.join(noise_only_root, task_name, demo_name)
            os.makedirs(save_dir_noised, exist_ok=True)
            os.makedirs(save_dir_noise_only, exist_ok=True)

            for i, frame in enumerate(demo_data):
                img_tensor = transform_to_tensor(Image.fromarray(frame)).unsqueeze(0)  # [1, 3, H, W]
                noise = model(img_tensor)              # [1, 3, H, W]
                noised_tensor = torch.clamp(img_tensor + noise, 0, 1)[0]

                # 保存噪声图像（原图+噪声）
                noised_image = transform_to_pil(noised_tensor)
                noised_image.save(os.path.join(save_dir_noised, f"frame_{i:03d}.png"))

                # 保存噪声本身（归一化到 [0, 1] 范围）
                noise_norm = (noise[0] + 1) / 2.0  # 原本 noise ∈ [-1,1]，归一化后 ∈ [0,1]
                noise_image = transform_to_pil(noise_norm.clamp(0, 1))
                noise_image.save(os.path.join(save_dir_noise_only, f"frame_{i:03d}.png"))


def encode_dataset(datas, labels, vla_encoder, siglip_processor, device='cpu'):
    transform_dino = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    features, targets = [], []

    for img_path, label in zip(datas, labels):
        image = Image.open(img_path).convert("RGB")

        image_tensor = transform_dino(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
        siglip_input = siglip_processor(images=image, return_tensors="pt").to(device)

        feat = vla_encoder(image_tensor, siglip_input)
        features.append(feat.squeeze(0).cpu())
        targets.append(label)

    return torch.stack(features), torch.tensor(targets)


def collect_image_paths_and_frame_counts(root_dir):
    image_paths = []
    frame_counts = []

    for task_name in sorted(os.listdir(root_dir)):
        task_dir = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_dir):
            continue

        for demo_subdir in sorted(os.listdir(task_dir)):
            if not demo_subdir.startswith("demo_"):
                continue

            demo_path = os.path.join(task_dir, demo_subdir)
            if not os.path.isdir(demo_path):
                continue

            demo_image_paths = []
            for file in sorted(os.listdir(demo_path)):
                if file.endswith(".png"):
                    full_path = os.path.join(demo_path, file)
                    demo_image_paths.append(full_path)
                    image_paths.append(full_path)

            frame_counts.append({
                "task": task_name,
                "demo": demo_subdir,
                "frames": len(demo_image_paths)
            })

    return image_paths, frame_counts
    
def extract_value_vectors(self, pixel_values: torch.Tensor, layer_ids: List[int] = None) -> Dict[str, List[torch.Tensor]]:
    """
    提取 transformer 中的 attention value 向量（V），不修改原 forward。
    返回结构：{ "layer_0": [v], "layer_1": [v], ... }
    """
    result = {}
    hook_handles = []
    value_dict = {}

    def register_hooks(blocks, name):
        for i, blk in enumerate(blocks):
            if (layer_ids is None) or (i in layer_ids):
                def hook_fn(module, input, output, layer_idx=i):
                    qkv = module.qkv(input[0])
                    q, k, v = qkv.chunk(3, dim=-1)
                    value_dict[f"layer_{layer_idx}"] = [v.detach()]
                h = blk.attn.register_forward_hook(hook_fn)
                hook_handles.append(h)

    if not self.use_fused_vision_backbone:
        register_hooks(self.featurizer.blocks, "featurizer")
        _ = self.forward(pixel_values)
        result = value_dict
    else:
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        register_hooks(self.featurizer.blocks, "featurizer")
        register_hooks(self.fused_featurizer.blocks, "fused_featurizer")
        _ = self.forward(pixel_values)
        result = value_dict

    for h in hook_handles:
        h.remove()

    return result
