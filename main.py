# -*- coding: utf-8 -*-
"""
Main script: Train perturbation generator with VLA model encoders.

新增能力（在不改变训练核心逻辑的前提下）：
- 运行隔离：每次运行使用独立 run 目录，便于追踪与清理。
- 丰富日志：run.log + args.json + env.json + TensorBoard（events/）。
- 完整参数快照：记录 args 的全部参数内容。
- 异常/中断清理：训练报错或中断会自动删除本次 run 的日志与权重文件。
- DataLoader 更稳健：num_workers 自适应与 pin_memory（若 CUDA）。
- 可复现性（可选）：若 args.seed 存在则固定随机种子。
"""

# ===========
# 标准库
# ===========
import os
import sys
import types
import random
import json
import shutil
import time
from datetime import datetime

# ===========
# 第三方库
# ===========
import numpy as np
from PIL import Image  # 可能在其他处用到，保留
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms

# ===========
# 项目内模块
# ===========
from args import args_parser
from utils import (
    DATE_TIME,
    extract_value_vectors,
    get_today_str,
    ensure_dir,
    copy_file_to_dir,
)
from datasets import DemoImageDataset
from VLAmodel import get_vla, get_processor
from autoadv_model import PerturbationGenerator
from train import train_epoch

from transformers import BertTokenizer, BertModel


# =========================================================
# 简单 transform：仅将 [0,255] HWC -> [0,1] CHW（保持原逻辑）
# =========================================================
def transform(image):
    """Convert PIL.Image or ndarray to torch.Tensor in [0,1], CHW."""
    to_tensor = transforms.ToTensor()
    return to_tensor(image)


# =========================
# 工具函数
# =========================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_remove(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.islink(path):
            os.unlink(path)
    except Exception as e:
        print(f"[WARN] 删除文件失败：{path} -> {e}")


def safe_rmtree(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f"[WARN] 删除目录失败：{path} -> {e}")


def log_line(fp, stage, msg):
    """将一行日志写入 run.log，并同步打印到控制台。"""
    line = f"[{now_str()}] [{stage}] {msg}"
    print(line)
    with open(fp, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_env_snapshot(device, num_workers, pin_memory):
    """采集当前环境信息用于记录。"""
    env = {
        "python": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "start_time": now_str(),
    }
    return env

def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """
    在可交互终端中询问用户，返回 True/False。
    - default=False 表示默认选择“否”（更安全，保留日志）。
    - 若非交互环境（stdin 非 TTY 或读入异常），直接返回 default。
    """
    try:
        if not sys.stdin or not sys.stdin.isatty():
            print(f"{prompt} [默认{'删除' if default else '保留'}，非交互环境]")
            return default
        suffix = " [y/N] " if not default else " [Y/n] "
        ans = input(prompt + suffix).strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        return default
    except Exception:
        return default



# =========================================================
# 入口
# =========================================================
if __name__ == '__main__':
    # -----------------------------
    # 1) 解析参数与随机种子（可复现性）
    # -----------------------------
    args = args_parser()
    today_str = get_today_str()

    if hasattr(args, "seed") and args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -----------------------------
    # 2) 目录组织：按 run_id 隔离
    # -----------------------------
    # 顶层保存目录：{log_path}/{dataset}/{suite}/
    base_save_path = os.path.join(args.log_path, args.dataset, args.VLA_task_suite)
    ensure_dir(base_save_path)

    run_id = f"EVAL-{args.VLA_task_suite}-{DATE_TIME}"
    run_dir = os.path.join(base_save_path, run_id)
    events_dir = os.path.join(run_dir, "events")
    images_dir = os.path.join(run_dir, "images")
    weights_dir = os.path.join(run_dir, "weights")
    backup_root_dir = os.path.join(run_dir, "codes")

    ensure_dir(run_dir)
    ensure_dir(events_dir)
    ensure_dir(images_dir)
    ensure_dir(weights_dir)
    ensure_dir(backup_root_dir)

    # 文本日志文件 & 参数/环境快照
    run_log_fp = os.path.join(run_dir, "run.log")
    args_json_fp = os.path.join(run_dir, "args.json")
    env_json_fp = os.path.join(run_dir, "env.json")

    # TensorBoard 目录
    logger = SummaryWriter(events_dir)

    # 为了便于异常清理，统一记录本 run 产生的“应删除对象”
    created_paths = set([run_dir, events_dir, images_dir, weights_dir, run_log_fp, args_json_fp, env_json_fp,backup_root_dir])

    # -----------------------------
    # 3) 备份关键源码文件（存在才复制）
    # -----------------------------
    for source_file in ['autoadv_model.py', 'train.py', 'main.py']:
        if os.path.exists(source_file):
            copy_file_to_dir(source_file, backup_root_dir)
        else:
            log_line(run_log_fp, "WARN", f"源文件不存在，跳过备份：{source_file}")
 
    # -----------------------------
    # 4) 数据集路径与 max_steps（确保 dataset_root 必有值）
    # -----------------------------
    dataset_root = getattr(args, "dataset_root", None)

    if args.dataset == 'libero':
        if args.VLA_task_suite == "libero_spatial":
            max_steps = 220
            dataset_root = "./datasets/libero"
        elif args.VLA_task_suite == "libero_object":
            max_steps = 280
            dataset_root = "./datasets/libero_object_no_noops"
        elif args.VLA_task_suite == "libero_goal":
            max_steps = 300
            # 使用外部 args.dataset_root 或在此给出错误
            dataset_root = "./datasets/libero_goal_no_noops"
        elif args.VLA_task_suite == "libero_10":
            max_steps = 520
            dataset_root = "./datasets/libero_10_no_noops"
        elif args.VLA_task_suite == "libero_90":
            max_steps = 400
            dataset_root = "./datasets/libero_90"

    if dataset_root is None:
        # 在日志中记录并抛异常，随后触发清理逻辑
        log_line(run_log_fp, "FATAL", "dataset_root 未设置。建议通过 --dataset_root 明确指定或在对应分支赋值。")
        # 清理
        logger.close()
        safe_rmtree(run_dir)
        sys.exit(1)

    # -----------------------------
    # 5) 设备选择 + 环境信息
    # -----------------------------
    device = torch.device(f'{args.device}' if torch.cuda.is_available() else 'cpu')
    # DataLoader 参数自适应
    default_workers_cap = 16
    cpu_count = os.cpu_count() or 4
    default_workers = min(cpu_count, default_workers_cap)
    num_workers = getattr(args, "num_workers", default_workers)
    pin_memory = (device.type == 'cuda')

    env_snapshot = get_env_snapshot(device, num_workers, pin_memory)
    dump_json(env_snapshot, env_json_fp)

    # -----------------------------
    # 6) 记录完整参数（args.json）与日志开头摘要
    # -----------------------------
    try:
        args_dict = dict(sorted(vars(args).items(), key=lambda x: x[0]))
    except Exception:
        # 某些 argparse.Namespace 可能含不可序列化对象，做兜底
        args_dict = {k: str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])}
    dump_json(args_dict, args_json_fp)

    log_line(run_log_fp, "START", f"Run ID: {run_id}")
    log_line(run_log_fp, "START", f"Dataset root: {dataset_root}")
    log_line(run_log_fp, "START", f"Device: {device}")
    log_line(run_log_fp, "START", f"Args snapshot -> {args_json_fp}")
    log_line(run_log_fp, "START", f"Env snapshot  -> {env_json_fp}")
    log_line(run_log_fp, "START", f"Events dir     -> {events_dir}")
    log_line(run_log_fp, "START", f"Images dir     -> {images_dir}")
    log_line(run_log_fp, "START", f"Weights dir    -> {weights_dir}")
    log_line(run_log_fp, "START", f"code dir    -> {backup_root_dir}")

    # -----------------------------
    # 7) 加载模型与处理器
    # -----------------------------
    try:
        VLAmodel = get_vla(args)
        visual_encoder = VLAmodel.vision_backbone
        visual_encoder.extract_value_vectors = types.MethodType(extract_value_vectors, visual_encoder)
        text_encoder = VLAmodel.language_model.model

        text_model = BertModel.from_pretrained(args.bert_path)  # 文本预训练模型
        text_model.to(device)


        processor = get_processor(args)
        log_line(run_log_fp, "INIT", "VLA 模型与处理器加载完成。")
    except Exception as e:
        log_line(run_log_fp, "FATAL", f"加载模型或处理器失败：{e}")
        logger.close()
        # 清理
        safe_rmtree(run_dir)
        sys.exit(1)

    # -----------------------------
    # 8) 数据集与 DataLoader
    # -----------------------------
    try:
        dataset = DemoImageDataset(args, args.VLA_task_suite, dataset_root)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        # 记录数据规模
        data_len = len(dataset) if hasattr(dataset, "__len__") else "unknown"
        log_line(run_log_fp, "DATA", f"Dataset size: {data_len}, batch_size={args.batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    except Exception as e:
        log_line(run_log_fp, "FATAL", f"加载数据集失败：{e}")
        logger.close()
        # 清理
        safe_rmtree(run_dir)
        sys.exit(1)

    # -----------------------------
    # 9) 初始化扰动生成器与优化器
    # -----------------------------
    perturbation_generator = PerturbationGenerator(input_channels=3, task_dim=4096).to(device)
    
    
    try:
        state_dict = torch.load(args.pre_train_weight, map_location=device)
        perturbation_generator.load_state_dict(state_dict)
        log_line(run_log_fp, "INIT", f"加载预训练权重：{args.pre_train_weight}")
    except Exception as e:
        log_line(run_log_fp, "WARN", f"加载预训练权重失败（继续训练）：{e}")

    optimizer = torch.optim.Adam(perturbation_generator.parameters(), lr=args.learn_rate)
    log_line(run_log_fp, "INIT", f"优化器就绪：Adam(lr={args.learn_rate})")

    # -----------------------------
    # 10) 训练循环
    # -----------------------------
    # 我们将 epoch 权重保存到 run_dir/weights/epoch_XXXX.pth
    # 另外保留你原先备份目录中的“当日最新权重”（会被覆盖），同时登记到 created_paths 便于失败清理。
    daily_weight_path = os.path.join(
        backup_root_dir,
        f'{args.VLA_task_suite}_perturbation_generator.pth'
    )
    created_paths.add(daily_weight_path)  # 失败时删除

    # 训练主体，带异常清理
    try:
        for epoch in range(args.epoch):
            t0 = time.time()
            results = train_epoch(
                processor=processor,
                dataloader=dataloader,
                optimizer=optimizer,
                visual_encoder=visual_encoder,
                text_encoder=text_encoder,
                bert = text_model,
                device=device,
                perturbation_generator=perturbation_generator,
                total_epoch=args.epoch,
                current_epoch=epoch,
                save_path=images_dir  # 将过程图像写入 run 专属目录
            )
            t1 = time.time()

                    # === 每 5 轮保存一次权重，最后一轮也保存 ===
            if (epoch + 1) % 5 == 0 or (epoch == args.epoch - 1):
                epoch_weight_path = os.path.join(weights_dir, f"epoch_{epoch+1:04d}.pth")
                torch.save(perturbation_generator.state_dict(), epoch_weight_path)
                created_paths.add(epoch_weight_path)

            # 保存/覆盖“当天备份目录”的最新权重
            torch.save(perturbation_generator.state_dict(), daily_weight_path)
            log_line(run_log_fp, "SAVE", f"保存权重: {os.path.basename(daily_weight_path)}")


            # 指标日志（文本 + TensorBoard）
            lr_list = [g.get("lr", None) if isinstance(g, dict) else None for g in getattr(optimizer, "param_groups", [])]
            lr_show = lr_list[0] if lr_list and lr_list[0] is not None else args.learn_rate

            mem_alloc = None
            mem_reserved = None
            if torch.cuda.is_available() and device.type == "cuda":
                try:
                    mem_alloc = torch.cuda.memory_allocated(device.index if device.index is not None else 0)
                    mem_reserved = torch.cuda.memory_reserved(device.index if device.index is not None else 0)
                except Exception:
                    pass

            metric_str = ", ".join([f"{k}: {v:.6f}" for k, v in results.items() if isinstance(v, (int, float))])
            extra_str = f"lr: {lr_show:.6e}, time: {t1 - t0:.2f}s"
            if mem_alloc is not None:
                extra_str += f", cuda_mem_alloc: {mem_alloc/1024/1024:.1f}MB, cuda_mem_reserved: {mem_reserved/1024/1024:.1f}MB"

            log_line(run_log_fp, "EPOCH",
                     f"({epoch+1}/{args.epoch}) {metric_str} | {extra_str} | ")

            # TensorBoard
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    logger.add_scalar(f"train/{k}", v, epoch)
            logger.add_scalar("train/lr", lr_show, epoch)
            logger.add_scalar("train/epoch_time_s", t1 - t0, epoch)

            log_line(run_log_fp, "EPOCH time_s",
                     f"({epoch+1}/{args.epoch}) {t1 - t0} |")

            if mem_alloc is not None:
                logger.add_scalar("train/cuda_memory_alloc_MB", mem_alloc / 1024 / 1024, epoch)
                logger.add_scalar("train/cuda_memory_reserved_MB", mem_reserved / 1024 / 1024, epoch)

        log_line(run_log_fp, "SUCCESS", "训练完成 ✅")

    except KeyboardInterrupt:
        log_line(run_log_fp, "INTERRUPT", "检测到手动中断（KeyboardInterrupt），开始清理本次运行产物。")
        try:
            logger.close()
        except Exception:
            pass

        delete_choice = ask_yes_no("检测到中断，是否删除本次运行的日志与保存的文件？（建议保留以便排查）", default=False)
        if delete_choice:
            print("[CLEANUP] 正在删除本次运行产物...")
            # 先删单个文件，再删整个 run 目录
            for p in list(created_paths):
                if os.path.isdir(p):
                    continue  # 留给最终 rmtree
                safe_remove(p)
            safe_rmtree(run_dir)
            print("[CLEANUP] 已删除本次运行产物。")
        else:
            print("[CLEANUP] 按照你的选择，保留本次运行的日志与文件。")
        sys.exit(1)

    except Exception as e:
        log_line(run_log_fp, "ERROR", f"训练过程中发生异常：{repr(e)}，开始清理本次运行产物。")
        logger.close()
        # 清理本 run 的文件与目录
        for p in list(created_paths):
            if os.path.isdir(p):
                continue  # 留给最终 rmtree
            safe_remove(p)
        safe_rmtree(run_dir)
        # 抛出异常以便外部感知（可按需改为 sys.exit(1)）
        raise

    finally:
        # 正常结束或异常/中断都会执行到这里
        try:
            logger.close()
        except Exception:
            pass

        # 正常完成训练时，保留 run_dir 作为完整记录；异常/中断时已提前清理。