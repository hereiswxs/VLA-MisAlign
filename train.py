import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from autoadv_model import PerturbationGenerator
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import os
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop, normalize
from utils import extract_value_vectors

import lpips
from transformers.image_processing_utils import BatchFeature
import math

import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel




# ---------- 小工具 ----------
def minmax_norm(x, eps=1e-6):
    B = x.size(0)
    xf = x.view(B, -1)
    minv = xf.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    maxv = xf.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    return (x - minv) / (maxv - minv + eps)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """计算 Total Variation (TV) 损失"""
    diff_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_y = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (diff_x.abs().mean() + diff_y.abs().mean())


class PerturbationRegularizationLoss(nn.Module):
    def __init__(self, lambda_l2: float = 1.0, lambda_tv: float = 0.1):
        """
        :param lambda_l2: L2 正则系数
        :param lambda_tv: TV 正则系数
        """
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_tv = lambda_tv

    def forward(self, perturbation: torch.Tensor):
        """
        :param perturbation: [B, C, H, W] 扰动张量
        :return: 损失值, 日志字典
        """
        # L2 正则（平均平方）
        loss_l2 = torch.mean(perturbation ** 2)

        # TV 正则（平滑性约束）
        loss_tv = tv_loss(perturbation)

        # 总损失
        loss = self.lambda_l2 * loss_l2 

        return 20*loss, {
            "loss_reg": loss.item(),
            "loss_l2": loss_l2.item(),
            "loss_tv": loss_tv.item()
        }


def schedule_epsilon_cosine(global_step: int,
                            total_steps: int,
                            eps_min: float = 0.05,
                            eps_max: float = 0.40,
                            warmup_ratio: float = 0.20,
                            ramp_ratio: float = 0.50):
    warmup_steps = int(total_steps * warmup_ratio)
    ramp_steps   = int(total_steps * ramp_ratio)
    if global_step <= warmup_steps:
        return eps_min
    prog = min((global_step - warmup_steps) / max(ramp_steps, 1), 1.0)  # 0→1
    # 0→1 映射到半个余弦周期：起步更缓，末端更平
    eased = 0.5 * (1.0 - math.cos(math.pi * prog))
    return eps_min + (eps_max - eps_min) * eased



class RelativeMisalignmentLoss(nn.Module):
    def __init__(self, 
                 vision_dim: int, 
                 text_dim: int, 
                 proj_dim: int = 1024, 
                 freeze_proj: bool = True,
                 mode: str = "relative",   # ["relative", "margin", "maxdiff"]
                 margin: float = 0.2):
        super().__init__()
        assert mode in ["relative", "margin", "maxdiff"], f"Unknown mode {mode}"
        self.mode = mode
        self.margin = margin

        self.vision_proj = nn.Linear(vision_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

        if freeze_proj:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, vision_feat_ori, vision_feat_pert, text_feat):
        # 投影并归一化
        v_ori = F.normalize(self.vision_proj(vision_feat_ori), dim=-1)
        v_pert = F.normalize(self.vision_proj(vision_feat_pert), dim=-1)
        t = F.normalize(self.text_proj(text_feat), dim=-1)

        # 计算相似度（cosine）
        sim_ori = (v_ori * t).sum(dim=-1)    # 原图-文本对齐度
        sim_pert = (v_pert * t).sum(dim=-1)  # 扰动图-文本对齐度

        # ----------- 损失定义 -----------
        if self.mode == "relative":
            # 要求 sim_pert <= sim_ori
            loss = torch.mean(F.relu(sim_pert - sim_ori))

        elif self.mode == "margin":
            # 要求 sim_pert + margin <= sim_ori
            loss = torch.mean(F.relu(sim_pert - (sim_ori - self.margin)))

        elif self.mode == "maxdiff":
            # 最大化 (sim_ori - sim_pert)，等价于最小化负差
            loss = -torch.mean(sim_ori - sim_pert)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # 返回损失和日志指标
        return loss, sim_pert.mean()



def encode_caption(tokenizer, caption):
    # 使用 BERT 将 caption 转换为特征表示
    
    inputs = tokenizer(caption,padding='max_length', 
                   max_length = 100, 
                   truncation=True,
                   return_tensors="pt")

    # 返回 BERT 特征向量
    return inputs  # 去掉 batch_size 维度



def train_epoch(processor, dataloader, optimizer, visual_encoder, text_encoder, bert,device,
                perturbation_generator, total_epoch, current_epoch, save_path,
                epsilon=0.4, gamma_lpips_relax=0.7):
    """
    关键改动：
      1) 生成器返回 (perturbation, M_task, freq_map)，构造 M = norm(M_task * freq_map)
      2) 特征差异：多层 token 差异做 “掩膜加权”，Gram 差异降权
      3) 扰动正则：加权 L1 + TV（对非目标区域增强惩罚）
      4) LPIPS：spatial=True，输入 normalize 到 [-1,1]，并用 (1 - γ M) 加权
      5) 文本-图像错配：沿用你的 RelativeMisalignmentLoss（把相似度推向 0）
    """
    visual_encoder.eval()
    text_encoder.eval()
    perturbation_generator.train()
    total_loss = total_feature_diff = total_penalty = total_alignment = total_lpips = total_ali = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch [{current_epoch}/{total_epoch}]")
    bert_tokenizer = BertTokenizer.from_pretrained('/home/student/DongXiaorong/MFAL/bert')

    layer_ids = [16, 18, 20, 22, 24]

    # LPIPS：拿空间图，并自动把输入从 [0,1] 归一到 [-1,1]
    loss_fn = lpips.LPIPS(net='alex', spatial=True).to(device)

    misalignment_loss_fn = RelativeMisalignmentLoss(
        vision_dim=2176,
        text_dim=4096,
        proj_dim=1024,
        mode="relative",   # 可选 "relative" / "margin" / "maxdiff"
    margin=0.1
    ).to(device)

    # 定义正则化 loss
    reg_loss_fn = PerturbationRegularizationLoss(lambda_l2=1.0, lambda_tv=0.1).to(device)



    for p in visual_encoder.parameters():
        p.requires_grad = False  # 冻结视觉编码器

    for step, batch in enumerate(progress_bar):
        images = batch["image"].to(device)  # [B,3,H,W], 约定在 [0,1]
        task_names = batch["task"]
        adv_txt_prompt = batch["adv_task"]

        # ------------------ 原图编码（无梯度） ------------------
        with torch.no_grad():
            #pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
            ori_inputs = processor(images=images, text=task_names,
                                   return_tensors="pt", padding=True).to(device, dtype=torch.float32)

            value_vectors_ori = visual_encoder.extract_value_vectors(
                ori_inputs["pixel_values"], layer_ids=layer_ids
            )  # dict: {"layer_16": (feat, ...), ...}
            original_features = visual_encoder(ori_inputs["pixel_values"])  # [B, D?]

        # ------------------ 文本编码（无梯度） ------------------

        with torch.no_grad():
            # 文本编码（无梯度）
            text_inputs = {k: v for k, v in ori_inputs.items() if k in ["input_ids", "attention_mask"]}
            task_outputs = text_encoder(**text_inputs)
            task_embed = task_outputs.last_hidden_state.mean(dim=1) # 取 [CLS] 向量作为全局语义

            advprompt_input = encode_caption(bert_tokenizer, adv_txt_prompt)

            advprompt_input = {key: value.to(device) for key, value in advprompt_input.items()}

            adv_task_outputs = bert(**advprompt_input)
            adv_task_embed = adv_task_outputs.last_hidden_state.mean(dim=1) # 取 [CLS] 向量作为全局语义

        # ------------------ 生成扰动 + 掩膜 ------------------
        # 现在生成器返回 (perturbation, M_task, freq_map)
        perturbation, task_cam, freq_cam = perturbation_generator(images, task_embed)
        #print(M_task.shape,freq_map.shape)
        # 计算全局步数（或你已有的 global_step）
        global_step = (current_epoch - 1) * len(dataloader) + step
        total_steps = total_epoch * len(dataloader)
        # 取得分步 epsilon
        epsilon_t = schedule_epsilon_cosine(global_step, total_steps,
                                            eps_min=0.35, eps_max=0.40,
                                            warmup_ratio=0.20, ramp_ratio=0.50)

        delta = 0.4 * perturbation
        perturbed_image = torch.clamp(images + delta, 0.0, 1.0)

        # ------------------ 扰动图编码 ------------------
        per_inputs = processor(images=perturbed_image, text=task_names,
                               return_tensors="pt", padding=True).to(device, dtype=torch.float32)
        value_vectors_pert = visual_encoder.extract_value_vectors(per_inputs["pixel_values"], layer_ids=layer_ids)
        perturbed_features = visual_encoder(per_inputs["pixel_values"])

        # ------------------ 特征差异（最大化，用负号） ------------------
        # 计算多层余弦差异
        layer_losses = []
        for layer in layer_ids:
            feat_ori = value_vectors_ori[f"layer_{layer}"][0]


            feat_pert = value_vectors_pert[f"layer_{layer}"][0]

            feat_ori = F.normalize(feat_ori, dim=-1)
            feat_pert = F.normalize(feat_pert, dim=-1)

            # ① token 分布差异（结构扰动）
            token_diff = F.mse_loss(feat_ori, feat_pert)  # 所有token结构扰动
            
            # ② Gram 矩阵差异（token间相对结构扰动）
            gram_ori = torch.bmm(feat_ori, feat_ori.transpose(1, 2)) / feat_ori.size(1)
            gram_pert = torch.bmm(feat_pert, feat_pert.transpose(1, 2)) / feat_pert.size(1)
            gram_diff = F.mse_loss(gram_ori, gram_pert)

            layer_loss = 0.5 * token_diff + 0.5 * gram_diff

            layer_losses.append(layer_loss)  # batch均值


        # 全局差异（与你原来一致）
        feature_dif = torch.mean(1 - F.cosine_similarity(perturbed_features, original_features, dim=1))

        # 组合：把“多层局部差异”（掩膜）和“全局差异”各占一半
        feature_diff = 0.2 * torch.stack(layer_losses).mean() + 0.8 * feature_dif

        # ------------------ 扰动正则（最小化）：加权 L1 + TV ------------------
        #w_off = (1.0 + beta_off * (1.0 - M)).expand_as(delta)  # 非目标处惩罚更重
        perturbation_penalty, logs_reg  = reg_loss_fn(perturbation)

        # ------------------ LPIPS（最小化）：空间加权 + 归一化 ------------------
        # loss_fn(..., spatial=True) 输出 [B,1,h,w]；normalize=True 自动把 [0,1] -> [-1,1]
        lpips_loss = loss_fn(perturbed_image, images, normalize=True).mean()  # [B,1,h,w]

        # ------------------ 文本-图像错配（最小化 |sim_pert|） ------------------
        # 这里沿用你的 RelativeMisalignmentLoss
        misalignment_loss, misalignment = misalignment_loss_fn(
            vision_feat_ori=original_features.mean(dim=1) if original_features.dim() == 3 else original_features,
            vision_feat_pert=perturbed_features.mean(dim=1) if perturbed_features.dim() == 3 else perturbed_features,
            text_feat=task_embed
        )

        # ------------------ 总 loss ------------------
        # 你的目标：最大化特征差异 → 用负号；其余项最小化
        loss = - feature_diff + perturbation_penalty + 10*lpips_loss + misalignment_loss

        # 反传
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(perturbation_generator.parameters(), 1.0)
        optimizer.step()

        # 统计
        total_loss       += loss.item()
        total_feature_diff += feature_diff.item()
        total_penalty    += perturbation_penalty.item()
        total_lpips      += lpips_loss.item()
        total_alignment  += misalignment_loss.item()
        total_ali        += misalignment.item()

        progress_bar.set_postfix({
            "loss": f"{total_loss/(step+1):.4f}",
            "cos_sim_diff": f"{(-total_feature_diff)/(step+1):.4f}",
            "penalty": f"{total_penalty/(step+1):.4f}",
            "lpips_loss": f"{total_lpips/(step+1):.4f}",
            "alignment_loss": f"{total_alignment/(step+1):.4f}",
            "alignment": f"{total_ali/(step+1):.4f}"
        })



        # 选步保存图像（可选）
        if step % 2 == 0:
            save_image(images[0].cpu(), os.path.join(save_path, "original.png"))
            save_image(perturbation[0].cpu(), os.path.join(save_path, "perturbation.png"))
            save_image(perturbed_image[0].cpu(), os.path.join(save_path, "perturbed.png"))
            save_image(freq_cam[0].cpu(), os.path.join(save_path, "freq_cam.png"))
            save_image(images[0].cpu() + freq_cam[0].cpu(), os.path.join(save_path, "aug_img.png"))
            #save_image(task_cam[0].cpu(), os.path.join(save_path, "task_cam.png"))


        
    # epoch统计打印
    avg_loss = total_loss / len(dataloader)
    avg_feature_diff = total_feature_diff / len(dataloader)
    avg_penalty = total_penalty / len(dataloader)
    avg_alignment = total_alignment / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)

    avg_ali = total_ali / len(dataloader)

    print(f"\n[Epoch {current_epoch}/{total_epoch}] "
          f"Avg Cosine Diff: {avg_feature_diff:.6f}, "
          f"Avg Penalty: {avg_penalty:.6f}, "
          f"avg_alignment: {avg_alignment:.6f}, "
          f"avg_lpips: {avg_lpips:.6f}, "
          f"Avg Loss: {avg_loss:.6f},"
          f"avg_ali: {avg_ali:.6f} ")

    return {
        "avg_loss": avg_loss,
        "avg_feature_diff": avg_feature_diff,
        "avg_penalty": avg_penalty,
        "avg_alignment": avg_alignment,
        "avg_lpips": avg_lpips,
        "avg_ali":avg_ali
    }