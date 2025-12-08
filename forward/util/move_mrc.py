import os
import subprocess
import glob
from tqdm import *


def copy_and_rename_mrc_files(source_dir, target_dir):
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 使用glob库查找所有.mrc文件
    mrc_files = glob.glob(os.path.join(source_dir, "*.mrc"))
    
    for i in trange(len(mrc_files)):
        source_file = mrc_files[i]
        # 获取文件名
        filename = os.path.basename(source_file)
        # 将文件名中的空格替换为下划线
        new_filename = filename.replace(" ", "_")
        # 构建目标文件的完整路径
        target_file = os.path.join(target_dir, new_filename[:-4]+'_part2.mrc')
        
        # 使用cp命令复制并重命名文件
        subprocess.run(['cp', source_file, target_file])
        # print(f"Copied and renamed: {source_file} -> {target_file}")


# 示例用法
source_directory = "/data/lifuwei/SPA_data/EMPIAR/10077-cn/data/Micrographs_part2/"
target_directory = "/data/lifuwei/SPA_data/EMPIAR-LFW/10077/mix/"
copy_and_rename_mrc_files(source_directory, target_directory)
