import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from PIL import Image
import argparse
import os

def visualize_clip_attention(image_path, text, output_dir, model_name="./clip_cache"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 CLIP 模型
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # 2. 加载图像
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 3. 提取视觉 patch 特征
    with torch.no_grad():
        vision_outputs = model.vision_model(inputs["pixel_values"])
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # [1, N, 768]

        # 投影到 512 维
        patch_tokens = model.visual_projection(patch_tokens)  # [1, N, 512]

        # 文本输入
        text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        text_outputs = model.text_model(**text_inputs)
        text_feat = text_outputs.last_hidden_state[:, 0, :]  # 取 CLS token [1, 512]
        text_embed = model.text_projection(text_feat)        # [1, 512]

    # 4. 计算相似度
    patch_tokens = F.normalize(patch_tokens, dim=-1)   # [1, N, 512]
    text_embed = F.normalize(text_embed, dim=-1)       # [1, 512]
    sim_map = torch.matmul(patch_tokens, text_embed.unsqueeze(-1)).squeeze(-1)  # [1, N]
    sim_map = sim_map.squeeze(0)  # 去掉 batch 维度，变成 [N]

    # 5. reshape 成空间热力图
    num_patches = sim_map.shape[0]
    grid_size = int(num_patches ** 0.5)  # e.g., 7x7 for ViT-B/32
    heatmap = sim_map.reshape(grid_size, grid_size).cpu().numpy()

    # 6. resize + 归一化
    heatmap = cv2.resize(heatmap, image.size)  # 插值到原图大小
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(np.array(image)[:, :, ::-1], 0.6, heatmap, 0.4, 0)

    # 7. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    orig_path = os.path.join(output_dir, "original.png")
    heatmap_path = os.path.join(output_dir, "clip_heatmap.png")
    overlay_path = os.path.join(output_dir, "overlay.png")

    image.save(orig_path)
    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(overlay_path, overlay)

    print(f"Saved results:\n - {orig_path}\n - {heatmap_path}\n - {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="/home/student/DongXiaorong/Madv_VLA/datasets/libero_object_no_noops/pick_up_the_ketchup_and_place_it_in_the_basket_demo/demo_7/frame_005.png", help="输入图像路径")
    parser.add_argument("--text", type=str, default="A robot arm is interacting with objects in the scene, and this image corresponds to the task: pick up the ketchup and place it in the basket", help="输入文本")
    parser.add_argument("--outdir", type=str, default="./clip_vis", help="保存目录")
    args = parser.parse_args()

    visualize_clip_attention(args.image, args.text, args.outdir)
