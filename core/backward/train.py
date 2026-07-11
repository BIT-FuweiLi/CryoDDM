import mrcfile as mf
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os
from tqdm import *
import torch.utils.data as data
import random
import argparse
import time
import sys
import torchvision.transforms as transforms
import copy

# 自动添加当前目录到系统路径，以便导入同级模块 (unet2d, util_self)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import unet2d
import util_self

def main(preprocess_path, model_save_path, gpus, batch_size, log_path):
    # 确保日志目录存在
    if log_path and not os.path.exists(log_path):
        os.makedirs(log_path)
    
    # 使用传入的 log_path，并格式化时间戳避免文件名非法字符
    # time.asctime() 可能包含空格和冒号，这里替换一下更安全
    time_str = str(time.asctime()).replace(' ', '_').replace(':', '-')
    temp = time_str + '.log'
    
    # 如果 log_path 为 None，默认存到模型保存路径下
    if log_path is None:
        log_path = model_save_path

    log_file = util_self.init_log(os.path.join(log_path, temp))
    
    log_file.write('Command: python ' + ' '.join(sys.argv) + '\n')
    log_file.write('reading data ~~\n')
    
    step1_path = os.path.join(preprocess_path, 's1')
    log_file.write('reading data from '+ step1_path +'\n')
    
    # 增加简单的文件存在性检查
    if not os.path.exists(step1_path):
        print(f"Error: Data path not found: {step1_path}")
        return

    s1_data = mf.read(os.path.join(step1_path, 'particles.mrcs'))
    s1_train_dataset = util_self.CustomDataset(s1_data, s1_data)
    s1_train_dataloader = torch.utils.data.DataLoader(s1_train_dataset, batch_size=batch_size, shuffle=True)

    step2_path = os.path.join(preprocess_path, 's2')
    log_file.write('reading data from '+step2_path+'\n')
    s2_input_data = mf.read(os.path.join(step2_path, 'input.mrcs'))
    s2_label_data = mf.read(os.path.join(step2_path, 'label.mrcs'))
    s2_train_dataset = util_self.CustomDataset(s2_input_data, s2_label_data)
    s2_train_dataloader = torch.utils.data.DataLoader(s2_train_dataset, batch_size=batch_size, shuffle=True)

    step3_path = os.path.join(preprocess_path, 's3')
    log_file.write('reading data from '+step3_path+'\n')
    s3_data = mf.read(os.path.join(step3_path, 'noise.mrcs'))
    s3_data_rand = copy.deepcopy(s3_data)
    np.random.shuffle(s3_data_rand)
    s3_train_dataset = util_self.CustomDataset(s3_data, s3_data_rand)
    s3_train_dataloader = torch.utils.data.DataLoader(s3_train_dataset, batch_size=batch_size, shuffle=True)

    val_path = os.path.join(preprocess_path, 'val')
    log_file.write('reading data from '+val_path+'\n')
    val_input_data = mf.read(os.path.join(val_path, 'input.mrcs'))
    val_label_data = mf.read(os.path.join(val_path, 'label.mrcs'))
    val_dataset = util_self.CustomDataset(val_input_data, val_label_data)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2*batch_size, shuffle=True)

    # 设置 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:{}'.format(gpus))
        model = unet2d.UDenoiseNet().cuda()
    else:
        print("Warning: CUDA not found, using CPU.")
        model = unet2d.UDenoiseNet()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    EPOCHES = 101
    s1_train_epochs_loss = []
    s2_train_epochs_loss = []
    s3_train_epochs_loss = []
    valid_epochs_loss = []
    log_file.write('training epoches ~~\n')

    # 早停机制参数
    # patience = 15
    # min_delta = 0.001
    # best_val_loss = float('inf')
    # counter = 0

    for epoch in tqdm(range(EPOCHES), file=log_file):
        model.train()
        s1_train_epoch_loss = []
        s2_train_epoch_loss = []
        s3_train_epoch_loss = []
        log_file.write('training batches ~~\n')

        # 训练 S1
        for batch_idx, (inputs, targets) in enumerate(s1_train_dataloader):
            if torch.cuda.is_available():
                inputs = inputs.unsqueeze(1).cuda()
                targets = targets.unsqueeze(1).cuda()
            else:
                inputs = inputs.unsqueeze(1)
                targets = targets.unsqueeze(1)
                
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            s1_train_epoch_loss.append(loss.item())
        
        # Scheduler 通常是在 epoch 结束时更新，或者 S1/S2/S3 全部跑完后更新一次
        # 原代码在每个 batch 循环里都写了 scheduler.step() 是不常见的用法
        # 但为了保持原逻辑不变，这里仅做缩进调整，或者你可以确认是否需要移到外层
        scheduler.step() 

        # 训练 S2
        for batch_idx, (inputs, targets) in enumerate(s2_train_dataloader):
            if torch.cuda.is_available():
                inputs = inputs.unsqueeze(1).cuda()
                targets = targets.unsqueeze(1).cuda()
            else:
                inputs = inputs.unsqueeze(1)
                targets = targets.unsqueeze(1)

            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            s2_train_epoch_loss.append(loss.item())

        # 训练 S3
        for batch_idx, (inputs, targets) in enumerate(s3_train_dataloader):
            if torch.cuda.is_available():
                inputs = inputs.unsqueeze(1).cuda()
                targets = targets.unsqueeze(1).cuda()
            else:
                inputs = inputs.unsqueeze(1)
                targets = targets.unsqueeze(1)

            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            s3_train_epoch_loss.append(loss.item())

        # 保存模型
        if (epoch+1) % 1 == 0:
            torch.save(model, os.path.join(model_save_path, str(epoch+1)+'.pth'))
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(model_save_path, 'checkpoint.pth'))

        s1_train_epochs_loss.append(np.average(s1_train_epoch_loss))
        s2_train_epochs_loss.append(np.average(s2_train_epoch_loss))
        s3_train_epochs_loss.append(np.average(s3_train_epoch_loss))
        log_file.write('epoch:'+str(epoch)+' s1_loss:'+str(s1_train_epochs_loss[-1])+' s2_loss:'+str(s2_train_epochs_loss[-1])+' s3_loss:'+str(s3_train_epochs_loss[-1])+'\n')

        # 验证
        if (epoch+1) % 1 == 0:
            log_file.write('valid modeling ...\n')
            model.eval()
            with torch.no_grad():
                valid_epoch_loss = []
                for idx, (inputs, labels) in enumerate(val_dataloader):
                    if torch.cuda.is_available():
                        inputs = inputs.unsqueeze(1).cuda()
                        labels = labels.unsqueeze(1).cuda()
                    else:
                        inputs = inputs.unsqueeze(1)
                        labels = labels.unsqueeze(1)
                        
                    outputs = model(inputs)
                    loss = criterion(labels, outputs)
                    valid_epoch_loss.append(loss.item())
                valid_epochs_loss.append(np.average(valid_epoch_loss))
            log_file.write('epoch:'+str(epoch)+' valid_loss:'+str(valid_epochs_loss[-1])+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3-step compress')
    parser.add_argument('--input_path', '-i', required=True, type=str, help='Preprocess data path (s1, s2, s3 folders)')
    parser.add_argument('--out_path', '-o', required=True, type=str, help='Model save path')
    parser.add_argument('--log_path', '-l', default=None, type=str, help='Log directory path')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--gpus', '-d', type=str, default='0', help='GPU ID')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # 如果没有提供 log_path，则默认在 out_path 下
    log_p = args.log_path if args.log_path else args.out_path

    main(args.input_path, args.out_path, args.gpus, args.batch_size, log_p)