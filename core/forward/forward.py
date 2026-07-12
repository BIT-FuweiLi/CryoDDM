import numpy as np
import mrcfile as mf
import os
from tqdm import *
import time
import argparse
from coordinates import normalize_xy, read_coordinate_file, resolve_coordinate_origin

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

def set_random_seed(seed):
    np.random.seed(int(seed))


def _jitter(shape, jitter_fraction):
    amount = int(jitter_fraction * shape)
    return 0 if amount <= 0 else np.random.randint(-amount, amount)


def _read_mrc(data_path, filename, cache, max_cache_size=10):
    basename = os.path.basename(filename)
    if basename not in cache:
        path = os.path.join(data_path, basename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"MRC file not found: {path}")
        data = mf.read(path)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D MRC image at {path}, got shape {data.shape}")
        if len(cache) >= max_cache_size:
            cache.pop(next(iter(cache)))
        cache[basename] = data
    return cache[basename]


def _crop_at_coordinate(mrc_data, record, shape, origin, jitter_fraction):
    row, col = normalize_xy(record, mrc_data.shape, origin)
    start_row = row - shape // 2 + _jitter(shape, jitter_fraction)
    start_col = col - shape // 2 + _jitter(shape, jitter_fraction)
    if start_row < 0 or start_col < 0:
        return None
    if start_row + shape > mrc_data.shape[0] or start_col + shape > mrc_data.shape[1]:
        return None
    patch = mrc_data[start_row:start_row + shape, start_col:start_col + shape]
    return patch if patch.shape == (shape, shape) else None


def get_particles_csstyle(data_path, star_path, shape, coord_origin="auto", jitter_fraction=0.2):
    """
    Extract particles from MRC files using STAR columns or legacy filename/x/y rows.
    """
    particles_mrc = []
    basename_count = {}
    mrc_cache = {}
    skipped = 0
    origin = resolve_coordinate_origin(star_path, coord_origin)
    for record in read_coordinate_file(star_path):
        mrc_file_name = os.path.basename(record.filename)
        if basename_count.get(mrc_file_name, 0) >= 20:
            continue
        mrc_data = _read_mrc(data_path, mrc_file_name, mrc_cache)
        patch = _crop_at_coordinate(mrc_data, record, shape, origin, jitter_fraction)
        if patch is None:
            skipped += 1
            continue
        particles_mrc.append(patch)
        basename_count[mrc_file_name] = basename_count.get(mrc_file_name, 0) + 1
        if len(particles_mrc) >= 4000:
            break
    if not particles_mrc:
        raise ValueError(
            f"No valid particle patches were extracted from {star_path}; "
            f"check coordinate origin, particle diameter, and image bounds. Skipped {skipped} entries."
        )
    return np.array(particles_mrc)

def get_noise(noise_label, in_path, shape, coord_origin="top-left"):
    """
    Retrieve noise patches from specified noise labels.
    """
    noise_patch = []
    cache = {}
    skipped = 0
    origin = resolve_coordinate_origin(noise_label, coord_origin)
    for record in read_coordinate_file(noise_label):
        temp = _read_mrc(in_path, record.filename, cache)
        noise = _crop_at_coordinate(temp, record, shape, origin, jitter_fraction=0)
        if noise is None:
            skipped += 1
            continue
        noise_patch.append(noise)
    if not noise_patch:
        raise ValueError(
            f"No valid noise patches were extracted from {noise_label}; "
            f"expected ({shape}, {shape}) patches. Skipped {skipped} entries."
        )
    return np.array(noise_patch)


def validate_diffusion_params(beta, total_steps, start):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite number greater than 0")
    if total_steps < 2:
        raise ValueError("total_steps must be at least 2")
    if start < 1 or start >= total_steps:
        raise ValueError("start must satisfy 1 <= start < total_steps")
    if beta * total_steps >= 1:
        raise ValueError("beta * total_steps must be less than 1")


def generate_diffusion_states(particles, noise, beta, total_steps):
    validate_diffusion_params(beta, total_steps, 1)
    if len(particles) == 0 or len(noise) == 0:
        raise ValueError("Particle and noise stacks must both be non-empty")
    if particles.shape != noise.shape:
        raise ValueError(f"Particle and noise stacks must have matching shapes, got {particles.shape} and {noise.shape}")
    states = [np.asarray(particles, dtype=np.float32)]
    noise_state = np.asarray(noise, dtype=np.float32).copy()
    for index in range(total_steps):
        alpha = 1 - beta * (index + 1)
        current = states[-1]
        residual = (np.sqrt(alpha) - 1) * current + np.sqrt(1 - alpha) * noise_state
        states.append(np.sqrt(alpha) * current + np.sqrt(1 - alpha) * residual)
        np.random.shuffle(noise_state)
    return states


def build_training_pairs(states, start):
    total_steps = len(states) - 1
    if total_steps < 2:
        raise ValueError("At least two generated diffusion steps are required")
    if start < 1 or start >= total_steps:
        raise ValueError("start must satisfy 1 <= start < total_steps")
    saved = states[start - 1:]
    train_pairs = [(saved[index + 1], saved[index]) for index in range(len(saved) - 2)]
    train_pairs.reverse()
    if not train_pairs:
        raise ValueError("The selected start/total_steps combination produces no S2 training pairs")
    train_input = np.vstack([pair[0] for pair in train_pairs])
    train_label = np.vstack([pair[1] for pair in train_pairs])
    return train_input, train_label, np.asarray(saved[-1]), np.asarray(saved[-2])

def get_diffuse_dataset_3step_compress_new(org_patch, noise_path, coordinate, 
                    shape, padding, save_path, log_file, isnormal=True, beta=0.1, total_steps=6, start=2,
                    particle_coord_origin="auto", noise_coord_origin="top-left"):
    """
    Generate diffuse dataset for three-step compression.
    """
    log_file.write('Time point: {} reading data\n'.format(time.asctime()))
    real_window = shape + 2*padding
    validate_diffusion_params(beta, total_steps, start)
    noise_patch = get_noise(noise_path, org_patch, real_window, coord_origin=noise_coord_origin)
    step3_path = os.path.join(save_path, 's3')
    log_file.write('\nSaving crops: {}\n'.format(step3_path))
    if not os.path.exists(step3_path):
        os.makedirs(step3_path)
    mf.write(os.path.join(step3_path,'noise.mrcs'), noise_patch.astype(np.float32), overwrite=True)
    particles_mrc = get_particles_csstyle(org_patch, coordinate, real_window, coord_origin=particle_coord_origin)
    if len(particles_mrc) <= len(noise_patch):
        random_index = np.random.choice(len(noise_patch), len(particles_mrc), replace=False)
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
    states = generate_diffusion_states(particles_mrc, noise_patch, beta, total_steps)
    for index in range(total_steps):
        alpha = 1 - beta * (index + 1)
        log_file.write('Step {}: Alpha = {:.4f}\n'.format(index + 1, alpha))
    train_input, train_label, val_input, val_label = build_training_pairs(states, start)
    mf.write(os.path.join(step2_path, 'input.mrcs'), train_input.astype(np.float32), overwrite=True)
    mf.write(os.path.join(step2_path, 'label.mrcs'), train_label.astype(np.float32), overwrite=True)
    val_path = os.path.join(save_path, 'val')
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    mf.write(os.path.join(val_path, 'input.mrcs'), val_input.astype(np.float32), overwrite=True)
    mf.write(os.path.join(val_path, 'label.mrcs'), val_label.astype(np.float32), overwrite=True)
    return True


def build_parser():
    parser = argparse.ArgumentParser(description='Forward process')
    parser.add_argument('--input_path', '-ip', default=None, type=str, help='Original mrc path')
    parser.add_argument('--particles_coordinate', '-pc', default=None, type=str, help='Particles coordinate .star file')
    parser.add_argument('--particle_diameter', '--particle_diamater', '-pd', dest='particle_diameter', default=200, type=int, help='Particle diameter in pixels')
    parser.add_argument('--noise_path', '-np', type=str, default=None, help='Noise label path')
    parser.add_argument('--out_path', '-op', type=str, default=None, help='Output path')
    parser.add_argument('--beta', type=float, default=0.1288, help='Beta value for diffusion')
    parser.add_argument('--total_steps', type=int, default=5, help='Total diffusion steps')
    parser.add_argument('--start', type=int, default=2, help='Start step for saving data')
    parser.add_argument('--particle_coord_origin', choices=('auto', 'top-left', 'bottom-left'), default='auto')
    parser.add_argument('--noise_coord_origin', choices=('top-left', 'bottom-left'), default='top-left')
    parser.add_argument('--seed', type=int, default=42)
    return parser

if __name__ == '__main__':
    args = build_parser().parse_args()
    validate_diffusion_params(args.beta, args.total_steps, args.start)
    set_random_seed(args.seed)
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # 构建命令
    command = [
        "python",
        __file__,
        "--input_path", args.input_path,
        "--particles_coordinate", args.particles_coordinate,
        "--particle_diameter", str(args.particle_diameter),
        "--noise_path", args.noise_path,
        "--out_path", args.out_path,
        "--beta", str(args.beta),
        "--total_steps", str(args.total_steps),
        "--start", str(args.start),
        "--particle_coord_origin", args.particle_coord_origin,
        "--noise_coord_origin", args.noise_coord_origin,
        "--seed", str(args.seed),
    ]

    # 初始化日志文件
    log_file_path = os.path.join(args.out_path, 'log.txt')
    log_file = open(log_file_path, 'w')

    # 将命令写入日志文件
    log_file.write('Executed command:\n')
    log_file.write(' '.join(command) + '\n\n')

    # 执行主逻辑
    aim_shape = int(args.particle_diameter * 1.5)
    aim_shape = int((aim_shape//128+1) * 128)
    padding = 0
    get_diffuse_dataset_3step_compress_new(org_patch=args.input_path, noise_path=args.noise_path, coordinate=args.particles_coordinate, shape=aim_shape, padding=padding, save_path=args.out_path, log_file=log_file, beta=args.beta, total_steps=args.total_steps, start=args.start, particle_coord_origin=args.particle_coord_origin, noise_coord_origin=args.noise_coord_origin)

    # 关闭日志文件
    log_file.close()
