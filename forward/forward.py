import numpy as np
import mrcfile as mf
import os
import glob
from tqdm import *
import time
import json
import sys
# sys.path.append('/data/lifuwei/small_protein/code/source_code4/forward')
import util_self
import argparse

def ensure_float32(arr):
    """
    Ensure that the input array is of type np.float32. If not, attempt to convert it to np.float32.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    if arr.dtype != np.float32:
        print("Warning: Automatically converting array type to np.float32")
        arr = arr.astype(np.float32)
    return arr

def get_particles_csstyle(data_path, star_path, shape):
    """
    Extract particles from MRC files based on coordinates provided in a CryoSPARC star file.
    The y-coordinate is adjusted to be relative to the bottom of the image.
    """
    particles_mrc = []
    basename_count = {}
    basename_limit = 4000
    basename_read = 0
    mrc_cache = {}
    max_cache_size = 10
    with open(star_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            columns = line.split()
            if len(columns) < 3:
                continue
            mrc_file_name = os.path.basename(columns[0])
            x_coord = float(columns[1])
            y_coord = float(columns[2])
            
            if mrc_file_name in basename_count and basename_count[mrc_file_name] >= 20:
                continue
            if mrc_file_name not in basename_count:
                if basename_read >= basename_limit:
                    continue
                basename_read += 1
            if mrc_file_name not in mrc_cache:
                mrc_path = os.path.join(data_path, mrc_file_name)
                mrc_data = mf.read(mrc_path)
                if len(mrc_cache) >= max_cache_size:
                    mrc_cache.pop(next(iter(mrc_cache)))
                mrc_cache[mrc_file_name] = mrc_data
            else:
                mrc_data = mrc_cache[mrc_file_name]
            
            # Adjust y-coordinate to be relative to the bottom of the image
            y_coord_adjusted = mrc_data.shape[1] - y_coord
            
            startx = int(x_coord) - shape // 2 + np.random.randint(int(-0.2 * shape), int(0.2 * shape))
            starty = int(y_coord_adjusted) - shape // 2 + np.random.randint(int(-0.2 * shape), int(0.2 * shape))
            
            if startx < 0 or startx + shape > mrc_data.shape[0] or starty < 0 or starty + shape > mrc_data.shape[1]:
                continue
            particles_mrc.append(mrc_data[startx:startx + shape, starty:starty + shape])
            if len(particles_mrc) >= 4000:
                break
            if mrc_file_name in basename_count:
                basename_count[mrc_file_name] += 1
            else:
                basename_count[mrc_file_name] = 1
    return np.array(particles_mrc)

def get_noise(noise_label, in_path, shape):
    """
    Retrieve noise patches from specified noise labels.
    """
    noise_patch = []
    noise_label = np.genfromtxt(noise_label, dtype=('U80', 'i4', 'i4'), encoding='utf-8')
    for i in range(len(noise_label)):
        temp = mf.read(os.path.join(in_path, noise_label[i][0]))
        noise = temp[noise_label[i][2]-int(shape//2):noise_label[i][2]+int(shape//2)+1,noise_label[i][1]-int(shape//2):noise_label[i][1]+int(shape//2)+1]
        noise = noise[:shape,:shape]
        if noise.shape!=(384,384):
            continue
        noise_patch.append(noise[:shape,:shape])
    # shapes = {n.shape for n in noise_patch}
    # if len(shapes) != 1:
    #     raise ValueError(f"Noise patch shapes are not consistent: {shapes}")
    return np.array(noise_patch)

def get_diffuse_dataset_3step_compress_new(org_patch, noise_path, coordinate, 
                    shape, padding, save_path, log_file, isnormal=True, beta=0.1, total_steps=6, start=2):
    """
    Generate diffuse dataset for three-step compression.
    """
    log_file.write('Time point: {} reading data\n'.format(time.asctime()))
    real_window = shape + 2*padding
    noise_patch = get_noise(noise_path, org_patch, real_window)
    step3_path = os.path.join(save_path, 's3')
    log_file.write('\nSaving crops: {}\n'.format(step3_path))
    if not os.path.exists(step3_path):
        os.makedirs(step3_path)
    mf.write(os.path.join(step3_path,'noise.mrcs'), noise_patch.astype(np.float32), overwrite=True)
    particles_mrc = get_particles_csstyle(org_patch, coordinate, real_window)
    if len(particles_mrc) <= len(noise_patch):
        random_index = np.random.choice(len(noise_patch), len(org_patch), replace=False)
        noise_patch = noise_patch[random_index]
    else:
        times = len(particles_mrc) // len(noise_patch)+1
        noise_patch = np.tile(noise_patch, (times,1,1))[:len(particles_mrc)]
        np.random.shuffle(noise_patch)
    log_file.write('Time point: {} diffusing\n'.format(time.asctime()))
    if isnormal:
        log_file.write('Diffuse by normal equation\n')
    else:
        log_file.write('Diffuse by org equation\n')
    step1_path = os.path.join(save_path, 's1')
    log_file.write('\nSaving crops: {}\n'.format(step1_path))
    if not os.path.exists(step1_path):
        os.makedirs(step1_path)
    mf.write(os.path.join(step1_path,'particles.mrcs'), particles_mrc.astype(np.float32), overwrite=True)
    step2_path = os.path.join(save_path, 's2')
    log_file.write('\nSaving crops: {}\n'.format(step2_path))
    if not os.path.exists(step2_path):
        os.makedirs(step2_path)
    inputs = []
    labels = []
    input_mrc = particles_mrc
    for i in range(total_steps):
        alpha = 1 - beta*(i+1)
        log_file.write('Step {}: Alpha = {:.4f}\n'.format(i+1, alpha))
        residual = (np.sqrt(alpha) - 1) * input_mrc + np.sqrt(1 - alpha) * noise_patch
        label_mrc = np.sqrt(alpha) * input_mrc + np.sqrt(1 - alpha) * residual
        if i+1 < start :
            log_file.write('Skip step {}: Alpha = {:.4f}\n'.format(i+1, alpha))
            np.random.shuffle(noise_patch)
            input_mrc = label_mrc
            continue
        else:
            log_file.write('Save step {}: Alpha = {:.4f}\n'.format(i+1, alpha))
            labels.append(label_mrc)
            inputs.append(input_mrc)
            np.random.shuffle(noise_patch)
            input_mrc = label_mrc
    mf.write(os.path.join(step2_path, 'input.mrcs'), np.vstack(inputs[:-1]).astype(np.float32), overwrite=True)
    mf.write(os.path.join(step2_path, 'label.mrcs'), np.vstack(labels[:-1]).astype(np.float32), overwrite=True)
    val_path = os.path.join(save_path, 'val')
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    mf.write(os.path.join(val_path, 'input.mrcs'), np.array(inputs[-1]).astype(np.float32), overwrite=True)
    mf.write(os.path.join(val_path, 'label.mrcs'), np.array(labels[-1]).astype(np.float32), overwrite=True)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forward process')
    parser.add_argument('--input_path', '-ip', default='path/to/dataset', type=str, help='Original mrc path')
    parser.add_argument('--particles_coordinate', '-pc', default="forward/10059/manualpick.star", type=str, help='Particles coordinate .star file')
    parser.add_argument('--particle_diamater', '-pd', default=200, type=int, help='Particle diameter in pixels')
    parser.add_argument('--noise_path', '-np', type=str, default='forward/10059/10059_.txt', help='Noise label path')
    parser.add_argument('--out_path', '-op', type=str, default='path/to/save', help='Output path')
    parser.add_argument('--beta', type=float, default=0.1288, help='Beta value for diffusion')
    parser.add_argument('--total_steps', type=int, default=5, help='Total diffusion steps')
    parser.add_argument('--start', type=int, default=2, help='Start step for saving data')
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # 构建命令
    command = [
        "python",
        __file__,
        "--input_path", args.input_path,
        "--particles_coordinate", args.particles_coordinate,
        "--particle_diamater", str(args.particle_diamater),
        "--noise_path", args.noise_path,
        "--out_path", args.out_path,
        "--beta", str(args.beta),
        "--total_steps", str(args.total_steps),
        "--start", str(args.start),
    ]

    # 初始化日志文件
    log_file_path = os.path.join(args.out_path, 'log.txt')
    log_file = open(log_file_path, 'w')

    # 将命令写入日志文件
    log_file.write('Executed command:\n')
    log_file.write(' '.join(command) + '\n\n')

    # 执行主逻辑
    aim_shape = int(args.particle_diamater * 1.5)
    aim_shape = int((aim_shape//128+1) * 128)
    padding = 0
    get_diffuse_dataset_3step_compress_new(org_patch=args.input_path, noise_path=args.noise_path, coordinate=args.particles_coordinate,  shape=aim_shape, padding=padding, save_path=args.out_path, log_file=log_file, beta=args.beta, total_steps=args.total_steps, start=args.start)

    # 关闭日志文件
    log_file.close()