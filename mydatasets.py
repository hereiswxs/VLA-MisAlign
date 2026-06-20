import os
from PIL import Image
import numpy as np

def load_libero_dataset(root_dir):
    """
    输入：root_dir，例如 '/home/student/DongXiaorong/Madv_VLA/datasets/libero'
    输出：
        labels: list of str，格式为 '任务名/demo编号'
        datas: list of np.array，每个数组形状为 (帧数, 高, 宽, 3)，类型为 uint8
    """
    labels = []
    datas = []

    for task_name in sorted(os.listdir(root_dir)):
        task_dir = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_dir):
            continue

        # 遍历 demo 子文件夹（如 demo_0、demo_1...）
        for demo_subdir in sorted(os.listdir(task_dir)):
            demo_path = os.path.join(task_dir, demo_subdir)
            if not os.path.isdir(demo_path):
                continue

            # 收集 PNG 图像
            img_files = [f for f in os.listdir(demo_path) if f.endswith('.png')]
            img_files.sort()

            if len(img_files) == 0:
                print(f"警告：{task_name}/{demo_subdir} 中无图像文件，跳过")
                continue

            frames = []
            for img_file in img_files:
                img_path = os.path.join(demo_path, img_file)
                img = Image.open(img_path).convert('RGB')
                frames.append(np.array(img))

            frames_array = np.stack(frames, axis=0)

            # 记录 label 和数据
            labels.append(f"{task_name}/{demo_subdir}")
            datas.append(frames_array)

    return labels, datas
