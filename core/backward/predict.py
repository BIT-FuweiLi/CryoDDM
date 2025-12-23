import mrcfile as mf
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os
from tqdm import trange, tqdm
import torch.utils.data as data
import random
import argparse
import torch.distributed as dist
import time
import unet2d
import sys
import util_self 
import torchvision.transforms as transforms
import copy 
import psutil  

def normal_batch(mrc_data):
    normal_data = np.zeros_like(mrc_data, dtype=np.float32)
    for i in range(len(mrc_data)):
        normal_data[i] = (mrc_data[i] - mrc_data[i].min()) / (mrc_data[i].max() - mrc_data[i].min())
    return normal_data

def normal(mrc_data):
    return (mrc_data - mrc_data.min()) / (mrc_data.max() - mrc_data.min())

def main(test_raw_path, test_out_path, test_model_path, gpu, aim_shape, log_dir):
    # 先读取测试文件
    log_file = util_self.init_log(os.path.join(log_dir, 'test_'+str(time.asctime())+'.log'))
    log_file.write(f"Command: {' '.join(sys.argv)}\n")
    
    test_list = util_self.get_rawdata_list(test_raw_path, log_file=log_file)
    output_list = util_self.get_rawdata_list(test_out_path, log_file=log_file)
    need_list = set(test_list) - set(output_list)
    test_list = list(need_list)
    
    if not test_list:
        print("No files to process.")
        sys.exit(0)
    
    first_file = test_list[0]
    first_data = np.array(mf.read(test_raw_path+'/'+first_file))
    y, x = first_data.shape
    
    # 计算单个文件的内存占用（假设数据类型为float32）
    single_file_size = x * y * 4  # 4 bytes per float32
    
    # 获取系统剩余内存
    mem = psutil.virtual_memory()
    available_memory = mem.available  # 剩余内存
    
    # 计算可以使用的内存（80% of available memory）
    usable_memory = available_memory * 0.8
    
    # 动态计算mrc_batch_size
    mrc_batch_size = int(usable_memory // single_file_size)
    if mrc_batch_size < 1:
        mrc_batch_size = 1  # 至少为1
    
    # 确保mrc_batch_size不超过文件总数
    if mrc_batch_size > len(test_list):
        mrc_batch_size = len(test_list)
    if mrc_batch_size > 500:
        mrc_batch_size = 500
    # 打印当前使用的mrc_batch_size
    log_file.write(f"Using mrc_batch_size: {mrc_batch_size}\n")
    
    num_batches = len(test_list) // mrc_batch_size + (1 if len(test_list) % mrc_batch_size != 0 else 0)
    padding = 64
    center_shape = [aim_shape-2*padding, aim_shape-2*padding]
    batch_size = 128
    device = torch.device('cuda:'+gpu)
    model = torch.load(test_model_path, map_location=device)
    
    for batch_num in trange(num_batches):
        mrc_data_np = []
        if batch_num + 1 < num_batches:
            for i in tqdm(range(mrc_batch_size), file=log_file):
                idx = mrc_batch_size * batch_num + i
                if idx >= len(test_list):
                    break
                filename = test_list[idx]
                log_file.write('reading {} ~~\n'.format(filename))
                mrc_data_np.append(np.array(mf.read(test_raw_path+'/'+filename)))
            test_data = normal_batch(mrc_data_np)
            for i in trange(len(mrc_data_np), file=log_file):
                filename = test_list[mrc_batch_size * batch_num + i]
                test_crops, sizes, matches = util_self.crop_data(test_data[i], center_shape, log_file=log_file, padding=padding, cval=0)
                
                all_output = []
                with torch.no_grad():
                    if len(test_crops) % batch_size == 0:
                        times = len(test_crops) // batch_size
                    else:
                        times = len(test_crops) // batch_size + 1 
                    for b in range(times):
                        inputs = torch.tensor(test_crops[b*batch_size:min((b+1)*batch_size,len(test_crops))], dtype=torch.float32).unsqueeze(1).to(device)
                        outputs = model(inputs)
                        all_output.append(outputs.squeeze(1).detach().cpu().numpy())
                
                all_output = np.vstack(all_output)
                mrc_out = util_self.concat_data(all_output, sizes, matches, log_file, padding)[:y, :x]
                mrc_out = normal(mrc_out)
                mf.write(os.path.join(test_out_path, filename), mrc_out.astype(np.float32), overwrite=True)
        else:
            remaining = len(test_list) - mrc_batch_size * batch_num
            for i in tqdm(range(remaining), file=log_file):
                idx = mrc_batch_size * batch_num + i
                filename = test_list[idx]
                log_file.write('reading {} ~~\n'.format(filename))
                mrc_data_np.append(np.array(mf.read(test_raw_path+'/'+filename)))
            test_data = normal_batch(mrc_data_np)
            for i in trange(remaining, file=log_file):
                filename = test_list[mrc_batch_size * batch_num + i]
                test_crops, sizes, matches = util_self.crop_data(test_data[i], center_shape, log_file=log_file, padding=padding, cval=0)
                
                all_output = []
                with torch.no_grad():
                    if len(test_crops) % batch_size == 0:
                        times = len(test_crops) // batch_size
                    else:
                        times = len(test_crops) // batch_size + 1 
                    for b in range(times):
                        inputs = torch.tensor(test_crops[b*batch_size:min((b+1)*batch_size,len(test_crops))], dtype=torch.float32).unsqueeze(1).to(device)
                        outputs = model(inputs)
                        all_output.append(outputs.squeeze(1).detach().cpu().numpy())
                
                all_output = np.vstack(all_output)
                mrc_out = util_self.concat_data(all_output, sizes, matches, log_file, padding)[:y, :x]
                mrc_out = normal(mrc_out)
                mf.write(os.path.join(test_out_path, filename), mrc_out.astype(np.float32), overwrite=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--input_path', '-i', type=str, default=None, help='raw_mrc_path')
    parser.add_argument('--out_path', '-o', type=str, default=None, help='save_path')
    parser.add_argument('--model_path', '-m', type=str, default=None, help='model')
    parser.add_argument('--particle_diamater', '-pd', type=int, default=200, help='particle_diamater')
    parser.add_argument('--gpus', '-d', type=str, default='0', help='gpus')
    parser.add_argument('--log_dir', '-l', type=str, default=None, help='log file directory')
    args = parser.parse_args()
    
    aim_shape = int(args.particle_diamater * 1.5)
    aim_shape = int((aim_shape//128+1) * 128)
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
     
    main(args.input_path, args.out_path, args.model_path, args.gpus,  aim_shape, args.log_dir) 