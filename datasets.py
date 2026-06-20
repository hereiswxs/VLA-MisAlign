from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T
import os
from glob import glob
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset_txt import libero_task_map
from transformers import BertTokenizer, BertModel

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

TASK_DESCRIPTION_PROMPT = (
    "Generate the smallest perturbation based on the task description, "
    "focusing on the object's edges and details to maximize the difference in image features, "
    "causing subtle changes that lead to task failure. "
    "The perturbation should have minimal impact on other regions to amplify the effect in key areas."
)





def grab_language_from_filename(x):
    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]

'''
                        # 获取图像文件名中的数字部分（如frame001，frame002等）
                        file_prefix = file_name.split('.')[0]  # 假设图像名格式为frame001.jpg, frame002.jpg
                        frame_number = int(file_prefix.split('_')[1])  # 提取数字部分，例如frame001中的001
if frame_number <=120:
                            image_path = os.path.join(demo_path, file_name)
                            self.samples.append((image_path, task_name, demo_name))
                            count += 1
'''

class DemoImageDataset(Dataset):
    def __init__(self, args, libero_suite, dataset_root, image_size = 256, transform=None):
        self.dataset_root = dataset_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为 Tensor
        ])
        self.samples = []  # list of (image_path, task_name, demo_name)

        task_list = os.listdir(dataset_root)

        for task in libero_task_map[libero_suite]:
            file_name = task + "_demo"
            assert file_name in task_list
            language = grab_language_from_filename(task + ".bddl")
            task_path = os.path.join(dataset_root, file_name)
            if not os.path.isdir(task_path):
                continue
            demo_list = os.listdir(task_path)
            for demo_name in demo_list:
                demo_path = os.path.join(task_path, demo_name)
                if not os.path.isdir(demo_path):
                    continue
                # 对于每个demo，只选择前80张图
                count = 0
                for file_name in sorted(os.listdir(demo_path)):
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(demo_path, file_name)
                        self.samples.append((image_path, language, demo_name))

        self.base_vla_name = args.VLA_path

            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, task_name, demo_name = self.samples[idx]
        image = Image.open(image_path).convert("RGB")


        if self.transform:
            image = self.transform(image)

        if "openvla-v01" in self.base_vla_name:  # OpenVLA v0.1
            prompt = (
                f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_name.lower()}? ASSISTANT:"
            )
        else:  
            prompt = f"In: What action should the robot take to {task_name.lower()}?\nOut:"

        advprompt = (
            f"A robot arm will '{task_name.lower()}'."
        )

        
        return {
            "image": image,
            "task": prompt,
            "adv_task": advprompt,
            "demo": demo_name,
            "path": image_path
        }





class LiberoDataset(Dataset):
    def __init__(self, samples, transform=None, num_frames=8):
        self.samples = samples  # 每个元素是 (label_str, [frame_path1, frame_path2, ...])
        self.transform = transform
        self.num_frames = num_frames
        
        # 1. 建立标签字符串到数字的映射
        all_labels = sorted(set(label for label, _ in samples))
        self.label2id = {label: idx for idx, label in enumerate(all_labels)}
        print("Label to ID map:", self.label2id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label_str, frame_paths = self.samples[idx]
        
        # 转换标签字符串为数字索引
        label = self.label2id[label_str]
        
        # 裁剪或补齐帧数
        if len(frame_paths) > self.num_frames:
            start = (len(frame_paths) - self.num_frames) // 2
            frame_paths = frame_paths[start:start + self.num_frames]
        elif len(frame_paths) < self.num_frames:
            pad_count = self.num_frames - len(frame_paths)
            frame_paths += [frame_paths[-1]] * pad_count
        
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)  # 一定要是Tensor类型
            else:
                img = T.ToTensor()(img)
            frames.append(img)
        video_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
        
        return torch.tensor(label, dtype=torch.long), video_tensor

def my_collate_fn(batch):
    labels, videos = zip(*batch)
    labels = torch.stack(labels)
    videos = torch.stack(videos)
    return labels, videos

def build_libero_samples(dataset_root, frame_ext='png', num_frames=120):
    samples = []
    
    # 每个任务目录 
    for task_name in os.listdir(dataset_root):
        task_path = os.path.join(dataset_root, task_name)
        if not os.path.isdir(task_path):
            continue
        
        # 每个 demo（比如 demo_0、demo_1）
        for demo_name in os.listdir(task_path):
            demo_path = os.path.join(task_path, demo_name)
            if not os.path.isdir(demo_path):
                continue

            # 找到所有帧图像
            frame_paths = sorted(glob(os.path.join(demo_path, f'frame_*.{frame_ext}')))
            if len(frame_paths) == 0:
                continue

            # 每条样本：任务名（字符串） + 帧路径列表
            samples.append((task_name, frame_paths[:num_frames]))  # 可按需裁剪帧数

    return samples
