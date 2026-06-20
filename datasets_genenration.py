import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
'''
#读取hd5文件中的数据
input_dir = "/home/student/DongXiaorong/openvla-main/LIBERO/libero/datasets/libero_spatial_no_noops"
output_dir = "/home/student/DongXiaorong/Madv_VLA/datasets/libero"

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    print(f"正在检查文件: {fname}")
    if fname.endswith('.hdf5') or fname.endswith('.h5'):
        h5_path = os.path.join(input_dir, fname)
        output_txt = os.path.join(output_dir, fname.replace('.hdf5', '.txt').replace('.h5', '.txt'))
        
        try:
            with h5py.File(h5_path, 'r') as f:
                content = []

                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        content.append(f"Dataset: {name} | shape: {obj.shape} | dtype: {obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        content.append(f"Group: {name}")

                f.visititems(visit)

                with open(output_txt, 'w') as out_f:
                    out_f.write('\n'.join(content))
                
                print(f"✔ 输出已保存至: {output_txt}")
        
        except Exception as e:
            print(f"❌ 读取失败: {h5_path}，原因: {e}")
'''

'''
input_folder = '/home/student/DongXiaorong/openvla-main/LIBERO/libero/datasets/libero_spatial_no_noops'
output_root = '/home/student/DongXiaorong/Madv_VLA/datasets/libero'
camera_view = 'agentview_rgb'

os.makedirs(output_root, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.hdf5') or filename.endswith('.h5'):
        file_path = os.path.join(input_folder, filename)
        print(f"正在处理文件: {file_path}")

        with h5py.File(file_path, 'r') as f:
            try:
                images = f[f'data/demo_30/obs/{camera_view}'][:]  # 形状 (N, H, W, 3)
            except KeyError as e:
                print(f"警告: 文件 {filename} 缺少图像数据，跳过。错误：{e}")
                continue

            base_name = os.path.splitext(filename)[0]
            save_dir = os.path.join(output_root, base_name)
            os.makedirs(save_dir, exist_ok=True)

            for idx, img in enumerate(images):
                pil_img = Image.fromarray(img)
                pil_img_rotated = pil_img.rotate(180)  # 逆时针旋转180度
                save_path = os.path.join(save_dir, f'frame_{idx:03d}.png')
                pil_img_rotated.save(save_path)

            print(f"已保存 {len(images)} 张图像至 {save_dir}")

print("全部文件处理完成！")
'''


import os
import h5py
from PIL import Image

input_folder = '/home/student/DongXiaorong/openvla-main/LIBERO/libero/datasets/libero_goal_no_noops'
output_root = '/home/student/DongXiaorong/Madv_VLA/datasets/libero_goal_no_noops'
camera_view = 'agentview_rgb'

os.makedirs(output_root, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.hdf5') or filename.endswith('.h5'):
        file_path = os.path.join(input_folder, filename)
        print(f"正在处理文件: {file_path}")

        with h5py.File(file_path, 'r') as f:
            demos = [key for key in f['data'].keys() if key.startswith('demo_')]
            
            for demo_key in demos:
                try:
                    images = f[f'data/{demo_key}/obs/{camera_view}'][:]
                except KeyError as e:
                    print(f"跳过 {filename} 中的 {demo_key}：{e}")
                    continue

                save_dir = os.path.join(output_root, os.path.splitext(filename)[0], demo_key)
                os.makedirs(save_dir, exist_ok=True)

                for idx, img in enumerate(images):
                    pil_img = Image.fromarray(img)
                    pil_img_rotated = pil_img.rotate(180)
                    save_path = os.path.join(save_dir, f'frame_{idx:03d}.png')
                    pil_img_rotated.save(save_path)

                print(f"已保存 {len(images)} 张图像至 {save_dir}")

print("全部文件处理完成！")



