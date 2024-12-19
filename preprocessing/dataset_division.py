import os
import random
import shutil

# 设置根路径
root_dir = 'path/to/CUGUV'

# 定义划分比例
split_ratio = {'train': 0.6, 'val': 0.2, 'test': 0.2}

# 遍历每个城市文件夹
for city in [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]:
    print(f"Processing city: {city}")
    
    # 获取影像和标签文件的路径
    img_path = os.path.join(root_dir, city, 'img')
    ann_path = os.path.join(root_dir, city, 'ann')
    
    # 确保img和ann文件夹存在并且非空
    if not os.path.exists(img_path) or not os.path.exists(ann_path):
        print(f"Missing img or ann folder in {city}. Skipping.")
        continue
    
    # 获取所有影像文件（假设与标签一一对应）
    files = [f for f in os.listdir(img_path) if f.endswith('.tif')]
    
    # 创建输出文件夹
    for subset in split_ratio.keys():
        os.makedirs(os.path.join(root_dir, city, 'img', subset), exist_ok=True)
        os.makedirs(os.path.join(root_dir, city, 'ann', subset), exist_ok=True)

    # 打乱文件顺序
    random.shuffle(files)
    
    # 计算分割点
    total_files = len(files)
    indices = list(range(total_files))
    train_split = int(total_files * split_ratio['train'])
    val_split = train_split + int(total_files * split_ratio['val'])

    # 分割数据集索引
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    # 定义复制函数
    def copy_files(indices, output_subset):
        for i in indices:
            file_name = files[i]
            img_file = os.path.join(img_path, file_name)
            ann_file = os.path.join(ann_path, file_name.replace('.tif', '.png'))
            
            if os.path.exists(img_file) and os.path.exists(ann_file):
                shutil.copy(img_file, os.path.join(root_dir, city, 'img', output_subset, file_name))
                shutil.copy(ann_file, os.path.join(root_dir, city, 'ann', output_subset, file_name.replace('.tif', '.png')))
            else:
                print(f"Missing file pair for {file_name} in {city}.")
    
    # 复制文件到相应的子集文件夹
    copy_files(train_indices, 'train')
    copy_files(val_indices, 'val')
    copy_files(test_indices, 'test')

print("Data splitting and copying completed.")

# 汇总数据集
output_img_dir = os.path.join(root_dir, 'img')
output_ann_dir = os.path.join(root_dir, 'ann')

# 创建输出文件夹
for subset in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_img_dir, subset), exist_ok=True)
    os.makedirs(os.path.join(output_ann_dir, subset), exist_ok=True)

# 定义复制并重命名函数
def copy_and_rename_files(city, subset):
    city_prefix = f"{city}_"
    img_source_dir = os.path.join(root_dir, city, 'img', subset)
    ann_source_dir = os.path.join(root_dir, city, 'ann', subset)
    
    if not os.path.exists(img_source_dir) or not os.path.exists(ann_source_dir):
        print(f"Missing img or ann folder for {city}/{subset}. Skipping.")
        return
    
    # 获取所有影像文件（假设与标签一一对应）
    files = [f for f in os.listdir(img_source_dir) if f.endswith('.tif')]
    
    for file_name in files:
        new_file_name = city_prefix + file_name
        img_src = os.path.join(img_source_dir, file_name)
        img_dst = os.path.join(output_img_dir, subset, new_file_name)
        
        ann_src = os.path.join(ann_source_dir, file_name.replace('.tif', '.png'))
        ann_dst = os.path.join(output_ann_dir, subset, new_file_name.replace('.tif', '.png'))
        
        # 确保源文件存在
        if os.path.exists(img_src) and os.path.exists(ann_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(ann_src, ann_dst)
        else:
            print(f"Missing file pair for {file_name} in {city}/{subset}.")

# 遍历每个城市文件夹
for city in [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]:
    print(f"Aggregating data from city: {city}")
    
    # 对每个子集进行处理
    for subset in ['train', 'val', 'test']:
        copy_and_rename_files(city, subset)

print("Data aggregation completed.")