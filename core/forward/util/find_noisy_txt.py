import argparse
import numpy as np
import os
import mrcfile as mf
import glob

def get_base_list(mrc_path, particle_path):
    """获取匹配的MRC和坐标文件基础名列表"""
    mrc_list = glob.glob(os.path.join(mrc_path, '*.mrc'))
    mrc_list = sorted(mrc_list)
    mrc_base = []
    
    # 获取MRC文件基础名
    for mrc_file in mrc_list:
        basename = os.path.basename(mrc_file)[:-4]
        mrc_base.append(basename)
    
    return mrc_base  # 返回所有MRC文件的基础名

def get_image_size(mrc_file):
    """只获取MRC文件的尺寸信息，不读取数据"""
    with open(mrc_file, 'rb') as f:
        # MRC文件头中的前三个4字节整数是图像尺寸
        f.seek(0)
        nx = int.from_bytes(f.read(4), byteorder='little')
        ny = int.from_bytes(f.read(4), byteorder='little')
    return (nx, ny)

def read_star_file(star_file):
    """读取star文件中的坐标信息"""
    coordinates = {}
    data_section = False
    
    with open(star_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            if line.startswith('data_'):
                data_section = True
                continue
                
            if line.startswith('loop_'):
                data_section = True
                continue
                
            if line.startswith('_'):  # 跳过列定义
                continue
                
            if data_section and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:  # 确保至少有文件名和xy坐标
                    try:
                        filename = os.path.splitext(parts[0])[0]  # 移除扩展名
                        x = int(float(parts[1]))
                        y = int(float(parts[2]))
                        
                        if filename not in coordinates:
                            coordinates[filename] = []
                        coordinates[filename].append([x, y])
                    except (ValueError, IndexError):
                        continue
    
    return coordinates

def does_overlap(new_center, box_size, existing_squares, max_points=100):
    """检查新方格是否与已有方格重叠"""
    if len(existing_squares) == 0:
        return False
        
    # 将坐标转换为numpy数组以加速计算
    if not isinstance(existing_squares, np.ndarray):
        existing_squares = np.array(existing_squares)
    
    # 计算新方格的边界
    new_x1 = new_center[0] - box_size / 2
    new_y1 = new_center[1] - box_size / 2
    new_x2 = new_center[0] + box_size / 2
    new_y2 = new_center[1] + box_size / 2
    
    # 计算所有已存在方格的边界
    x1 = existing_squares[:, 0] - box_size / 2
    y1 = existing_squares[:, 1] - box_size / 2
    x2 = existing_squares[:, 0] + box_size / 2
    y2 = existing_squares[:, 1] + box_size / 2
    
    # 向量化重叠检查
    overlaps = (new_x1 < x2) & (new_x2 > x1) & (new_y1 < y2) & (new_y2 > y1)
    return np.any(overlaps)

def find_noise_coordinates(canvas_size, box_size, existing_squares, max_points=100):
    """找到所有可能的噪声区域中心点"""
    noise_points = []
    margin = 200  # 边缘留白
    step = box_size // 8  # 增大步长
    
    # 创建网格
    x_range = range(box_size//2 + margin, canvas_size[0] - box_size//2 - margin, step)
    y_range = range(box_size//2 + margin, canvas_size[1] - box_size//2 - margin, step)
    
    # 随机打乱网格点
    grid_points = [(x, y) for x in x_range for y in y_range]
    np.random.shuffle(grid_points)
    
    # 只检查部分点直到找到足够的噪声点
    for x, y in grid_points:
        if len(noise_points) >= max_points:
            break
            
        if not does_overlap((x, y), box_size, existing_squares) and not does_overlap((x, y), box_size, noise_points):
            noise_points.append([x, y])
    
    return noise_points

def main(args):
    """主函数,只生成噪声坐标"""
    # 确保输出文件的目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有MRC文件的基础名
    base_list = get_base_list(args.mrc_path, args.label_path)
    
    # 读取star文件中的坐标信息
    particle_coords = read_star_file(args.label_path)
    
    # 清空或创建坐标文件
    with open(args.output_path, 'w') as f:
        pass  # 只清空文件，不写入表头
    
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 预先获取所有MRC文件的尺寸
    print("Reading image sizes...")
    image_sizes = {}
    for basename in base_list:
        mrc_file = os.path.join(args.mrc_path, basename + '.mrc')
        image_sizes[basename] = get_image_size(mrc_file)
    
    print("Processing coordinates...")
    for basename in base_list:
        # 使用预先读取的尺寸
        canvas_size = image_sizes[basename]
        
        # 获取该文件的颗粒坐标
        coords = np.array(particle_coords.get(basename, []), dtype=np.int32)
        if len(coords) == 0:
            print(f"Warning: No coordinates found for {basename}")
            continue
        
        # 寻找噪声区域的中心点
        noise_coords = find_noise_coordinates(
            canvas_size=canvas_size,
            box_size=args.box_size,
            existing_squares=coords,
            max_points=100  # 限制每个图像的噪声点数量
        )

        # 将坐标添加到文件中
        with open(args.output_path, 'a') as f:
            for coord in noise_coords:
                f.write(f"{basename}.mrc {coord[0]} {coord[1]}\n")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Generate noise coordinates from MRC files')
    parser.add_argument('-i', '--mrc_path', required=True,
                        help='Path to MRC files directory')
    parser.add_argument('-l', '--label_path', required=True,
                        help='Path to label/coordinate files directory')
    parser.add_argument('-b', '--box_size', type=int, required=True,
                        help='Box size for extraction')
    parser.add_argument('-o', '--output_path', required=True,
                        help='Path to output coordinates file (e.g., /path/to/noise_coords.txt)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
