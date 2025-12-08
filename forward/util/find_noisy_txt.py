import argparse
import numpy as np
import os
import mrcfile as mf
import glob

def get_base_list(mrc_path, particle_path):
    """获取匹配的MRC和坐标文件基础名列表"""
    mrc_list = glob.glob(os.path.join(mrc_path, '*.mrc'))
    particle_list = glob.glob(os.path.join(particle_path, '*.txt'))

    mrc_list = sorted(mrc_list)
    mrc_base = []
    particle_base = []
    
    # 获取MRC文件基础名
    for mrc_file in mrc_list:
        basename = os.path.basename(mrc_file)[:-4]
        mrc_base.append(basename)
    
    # 获取坐标文件基础名
    for particle_file in particle_list:
        basename = os.path.basename(particle_file)[:-4]
        particle_base.append(basename)
    
    # 找到同时存在的文件
    new_list = []
    for basename in mrc_base:
        if basename in particle_base:
            new_list.append(basename + ".mrc")
    return new_list

def get_paired_data(mrc_path, particle_path, basename):
    """获取配对的MRC数据和坐标数据"""
    # 检查basename是否已经包含.mrc后缀
    if basename.endswith('.mrc'):
        basename = basename[:-4]

    mrc_file = os.path.join(mrc_path, basename + '.mrc')
    particle_file = os.path.join(particle_path, basename + '.txt')

    # 读取MRC数据和坐标数据
    mrc_data = mf.read(mrc_file)
    particle_data = np.genfromtxt(particle_file)
    if particle_data.ndim == 1:
        particle_data = np.expand_dims(particle_data, axis=0)
    particle_data = particle_data.astype(np.int32)
    return mrc_data, particle_data

def does_overlap(new_center, box_size, existing_squares):
    """检查新方格是否与已有方格重叠"""
    new_x1 = new_center[0] - box_size / 2
    new_y1 = new_center[1] - box_size / 2
    new_x2 = new_center[0] + box_size / 2
    new_y2 = new_center[1] + box_size / 2

    for center in existing_squares:
        x1 = center[0] - box_size / 2
        y1 = center[1] - box_size / 2
        x2 = center[0] + box_size / 2
        y2 = center[1] + box_size / 2

        if new_x1 < x2 and new_x2 > x1 and new_y1 < y2 and new_y2 > y1:
            return True
    return False

def find_space_for_new_square(canvas_size, box_size, existing_squares):
    """在画布上找到可以放置新方格的位置"""
    noise_points = []
    margin = 500  # 边缘留白
    step = box_size // 8  # 搜索步长
    
    for x in range(box_size//2 + margin, canvas_size[0] - box_size//2 - margin, step):
        for y in range(box_size//2 + margin, canvas_size[1] - box_size//2 - margin, step):
            if not does_overlap((x, y), box_size, existing_squares) and not does_overlap((x, y), box_size, noise_points):
                noise_points.append([x, y])
    return noise_points

def main(args):
    """主函数,处理噪声提取"""
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    txt_path = os.path.join(args.output_path, "coordinates")
    os.makedirs(txt_path, exist_ok=True)

    # 获取基础文件列表
    base_list = get_base_list(args.mrc_path, args.label_path)
    
    for base_name in base_list:
        # 获取配对的数据
        mrc_data, particle_data = get_paired_data(args.mrc_path, args.label_path, base_name)
        particle_data = particle_data[:, :2]  # 只使用坐标信息
        
        # 寻找噪声区域的中心点
        centers = find_space_for_new_square(
            canvas_size=mrc_data.shape,
            box_size=args.box_size,
            existing_squares=particle_data
        )

        # 裁剪并保存噪声区域
        for i, center in enumerate(centers):
            x_start = center[0] - args.box_size // 2
            x_end = center[0] + args.box_size // 2
            y_start = center[1] - args.box_size // 2
            y_end = center[1] + args.box_size // 2
            
            # 确保索引在有效范围内
            if (x_start >= 0 and x_end <= mrc_data.shape[0] and 
                y_start >= 0 and y_end <= mrc_data.shape[1]):
                
                cropped_data = mrc_data[x_start:x_end, y_start:y_end]
                
                # 保存裁剪的MRC文件
                basename = os.path.splitext(base_name)[0]
                output_mrc = os.path.join(args.output_path, f"{basename}_noise_{i}.mrc")
                mf.write(output_mrc, cropped_data, overwrite=True)
                
                # 保存坐标信息
                coord_file = os.path.join(txt_path, f"{basename}_coordinates.txt")
                with open(coord_file, 'a') as f:
                    f.write(f"{center[0]}, {center[1]}\n")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Extract noise regions from MRC files')
    parser.add_argument('-i', '--mrc_path', required=True,
                        help='Path to MRC files directory')
    parser.add_argument('-l', '--label_path', required=True,
                        help='Path to label/coordinate files directory')
    parser.add_argument('-b', '--box_size', type=int, required=True,
                        help='Box size for extraction')
    parser.add_argument('-o', '--output_path', required=True,
                        help='Output directory path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
