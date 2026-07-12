import mrcfile as mf
import torch
import numpy as np
import os
from tqdm import trange, tqdm
import argparse
import time
import unet2d
import sys
import util_self 
import psutil  

def _normalize_array(mrc_data):
    data = np.asarray(mrc_data, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("MRC data contains NaN or Inf values")
    minimum = float(data.min())
    data_range = float(data.max()) - minimum
    if data_range == 0:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - minimum) / data_range).astype(np.float32, copy=False)


def normal_batch(mrc_data):
    return [_normalize_array(image) for image in mrc_data]

def normal(mrc_data):
    return _normalize_array(mrc_data)

def calculate_mrc_batch_size(available_memory, single_file_size, file_count):
    if single_file_size <= 0:
        raise ValueError("single_file_size must be positive")
    if file_count <= 0:
        return 1
    estimated = int((available_memory * 0.2) // single_file_size)
    return max(1, min(estimated, file_count, 64))


def plan_mrc_batches(filenames, available_memory, size_lookup, max_files=64):
    budget = max(1, int(available_memory * 0.2))
    batches = []
    current = []
    current_bytes = 0
    for filename in filenames:
        estimated_bytes = max(1, int(size_lookup(filename)))
        if current and (len(current) >= max_files or current_bytes + estimated_bytes > budget):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(filename)
        current_bytes += estimated_bytes
    if current:
        batches.append(current)
    return batches


def estimate_mrc_memory(path):
    with mf.mmap(os.fspath(path), mode='r', permissive=True) as mrc:
        data = mrc.data
        return max(int(data.nbytes), int(np.prod(data.shape)) * 4)


def main(test_raw_path, test_out_path, test_model_path, gpu, aim_shape, log_dir, batch_size=16):
    test_raw_path = os.fspath(test_raw_path)
    test_out_path = os.fspath(test_out_path)
    log_dir = os.fspath(log_dir)
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    log_file = util_self.init_log(os.path.join(log_dir, 'test_'+str(time.asctime())+'.log'))
    try:
        log_file.write(f"Command: {' '.join(sys.argv)}\n")
        test_list = util_self.get_rawdata_list(test_raw_path, log_file=log_file)
        output_list = util_self.get_rawdata_list(test_out_path, log_file=log_file)
        test_list = sorted(set(test_list) - set(output_list))
        if not test_list:
            print("No files to process.")
            return

        available_memory = psutil.virtual_memory().available
        file_batches = plan_mrc_batches(
            test_list,
            available_memory,
            lambda filename: estimate_mrc_memory(os.path.join(test_raw_path, filename)),
        )
        log_file.write(f"Using {len(file_batches)} MRC batch(es), max 64 files and 20% memory budget\n")

        padding = 64
        center_shape = [aim_shape-2*padding, aim_shape-2*padding]
        device = torch.device('cuda:'+gpu)
        model = torch.load(test_model_path, map_location=device)
        model.eval()

        for filenames in tqdm(file_batches, file=log_file):
            raw_images = []
            for filename in filenames:
                log_file.write('reading {} ~~\n'.format(filename))
                raw_images.append(np.asarray(mf.read(os.path.join(test_raw_path, filename))))
            normalized_images = normal_batch(raw_images)
            for filename, raw_image, normalized_image in zip(filenames, raw_images, normalized_images):
                image_y, image_x = raw_image.shape
                test_crops, sizes, matches = util_self.crop_data(
                    normalized_image, center_shape, log_file=log_file, padding=padding, cval=0
                )
                all_output = []
                with torch.no_grad():
                    for offset in range(0, len(test_crops), batch_size):
                        inputs = torch.tensor(
                            test_crops[offset:offset + batch_size], dtype=torch.float32
                        ).unsqueeze(1).to(device)
                        outputs = model(inputs)
                        all_output.append(outputs.squeeze(1).detach().cpu().numpy())
                reconstructed = util_self.concat_data(
                    np.vstack(all_output), sizes, matches, log_file, padding
                )[:image_y, :image_x]
                mf.write(
                    os.path.join(test_out_path, filename),
                    normal(reconstructed).astype(np.float32),
                    overwrite=True,
                )
    finally:
        log_file.close()

def build_parser():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--input_path', '-i', type=str, default=None, help='raw_mrc_path')
    parser.add_argument('--out_path', '-o', type=str, default=None, help='save_path')
    parser.add_argument('--model_path', '-m', type=str, default=None, help='model')
    parser.add_argument('--particle_diameter', '--particle_diamater', '-pd', dest='particle_diameter', type=int, default=200, help='particle diameter in pixels')
    parser.add_argument('--batch_size', '-bs', type=int, default=16, help='crop batch size for inference')
    parser.add_argument('--gpus', '-d', type=str, default='0', help='gpus')
    parser.add_argument('--log_dir', '-l', type=str, default=None, help='log file directory')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    
    aim_shape = int(args.particle_diameter * 1.5)
    aim_shape = int((aim_shape//128+1) * 128)
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
     
    main(args.input_path, args.out_path, args.model_path, args.gpus, aim_shape, args.log_dir, batch_size=args.batch_size)
