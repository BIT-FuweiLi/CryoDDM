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
import copy

# 自动添加当前目录到系统路径，以便导入同级模块 (unet2d, util_self)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import unet2d
import util_self
from best_model_selection import BEST_MODEL_RULE, materialize_best_model

REQUIRED_TRAINING_FILES = {
    "s1": "s1/particles.mrcs",
    "s2_input": "s2/input.mrcs",
    "s2_label": "s2/label.mrcs",
    "s3": "s3/noise.mrcs",
    "val_input": "val/input.mrcs",
    "val_label": "val/label.mrcs",
}


def _as_stack(array, label):
    stack = np.asarray(array)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    if stack.ndim != 3 or len(stack) == 0:
        raise ValueError(f"Training data {label} must be a non-empty 3D stack, got shape {stack.shape}")
    return stack


def validate_training_data(preprocess_path):
    root = os.fspath(preprocess_path)
    loaded = {}
    for label, relative_path in REQUIRED_TRAINING_FILES.items():
        path = os.path.join(root, relative_path)
        if not os.path.isfile(path):
            raise ValueError(f"Missing required training file: {path}")
        loaded[label] = _as_stack(mf.read(path), label)
    for prefix in ("s2", "val"):
        inputs = loaded[f"{prefix}_input"]
        labels = loaded[f"{prefix}_label"]
        if len(inputs) != len(labels):
            raise ValueError(f"{prefix} input and label stacks must contain the same number of images")
        if inputs.shape[1:] != labels.shape[1:]:
            raise ValueError(f"{prefix} input and label patch shapes must match")
    return loaded


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_mean(losses, label, log_file):
    value = float(np.average(losses))
    if not np.isfinite(value):
        message = f"Training produced a non-finite {label}: {value}"
        log_file.write(message + '\n')
        log_file.close()
        raise RuntimeError(message)
    return value


def main(preprocess_path, model_save_path, gpus, batch_size, log_path, epochs=101, seed=42):
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    os.makedirs(model_save_path, exist_ok=True)
    stale_best_model = os.path.join(model_save_path, 'best_model.pth')
    if os.path.exists(stale_best_model):
        os.remove(stale_best_model)
    datasets = validate_training_data(preprocess_path)
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
    
    set_random_seed(seed)

    step1_path = os.path.join(preprocess_path, 's1')
    log_file.write('reading data from '+ step1_path +'\n')
    
    s1_data = datasets["s1"]
    s1_train_dataset = util_self.CustomDataset(s1_data, s1_data)
    s1_train_dataloader = torch.utils.data.DataLoader(s1_train_dataset, batch_size=batch_size, shuffle=True)

    step2_path = os.path.join(preprocess_path, 's2')
    log_file.write('reading data from '+step2_path+'\n')
    s2_input_data = datasets["s2_input"]
    s2_label_data = datasets["s2_label"]
    s2_train_dataset = util_self.CustomDataset(s2_input_data, s2_label_data)
    s2_train_dataloader = torch.utils.data.DataLoader(s2_train_dataset, batch_size=batch_size, shuffle=True)

    step3_path = os.path.join(preprocess_path, 's3')
    log_file.write('reading data from '+step3_path+'\n')
    s3_data = datasets["s3"]
    s3_data_rand = copy.deepcopy(s3_data)
    np.random.shuffle(s3_data_rand)
    s3_train_dataset = util_self.CustomDataset(s3_data, s3_data_rand)
    s3_train_dataloader = torch.utils.data.DataLoader(s3_train_dataset, batch_size=batch_size, shuffle=True)

    val_path = os.path.join(preprocess_path, 'val')
    log_file.write('reading data from '+val_path+'\n')
    val_input_data = datasets["val_input"]
    val_label_data = datasets["val_label"]
    val_dataset = util_self.CustomDataset(val_input_data, val_label_data)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2*batch_size, shuffle=False)

    # 设置 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:{}'.format(gpus))
        model = unet2d.UDenoiseNet().cuda()
    else:
        print("Warning: CUDA not found, using CPU.")
        model = unet2d.UDenoiseNet()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

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

    loss_records = []
    for epoch in tqdm(range(epochs), file=log_file):
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
            s3_train_epoch_loss.append(loss.item())

        scheduler.step()

        s1_loss = _finite_mean(s1_train_epoch_loss, 'S1 loss', log_file)
        s2_loss = _finite_mean(s2_train_epoch_loss, 'S2 loss', log_file)
        s3_loss = _finite_mean(s3_train_epoch_loss, 'S3 loss', log_file)

        # 验证
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
        valid_loss = _finite_mean(valid_epoch_loss, 'validation loss', log_file)

        torch.save(model, os.path.join(model_save_path, str(epoch+1)+'.pth'))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(model_save_path, 'checkpoint.pth'))

        s1_train_epochs_loss.append(s1_loss)
        s2_train_epochs_loss.append(s2_loss)
        s3_train_epochs_loss.append(s3_loss)
        valid_epochs_loss.append(valid_loss)
        loss_records.append({"epoch": epoch + 1, "s2_loss": s2_loss})
        best = materialize_best_model(loss_records, model_save_path)
        if best is not None:
            log_file.write(
                'best_model epoch:{} s2_loss:{} rule:{}\n'.format(
                    best['epoch'], best['s2_loss'], BEST_MODEL_RULE
                )
            )
        log_file.write(
            'epoch:{} s1_loss:{} s2_loss:{} s3_loss:{} valid_loss:{}\n'.format(
                epoch, s1_loss, s2_loss, s3_loss, valid_loss
            )
        )

    if not os.path.exists(os.path.join(model_save_path, 'best_model.pth')):
        log_file.write(
            'No stable best model: no epoch > 20 had 10 following epochs within the 10% s2_loss stability rule. '
            'Choose a specific epoch .pth for prediction.\n'
        )
    log_file.close()


def build_parser():
    parser = argparse.ArgumentParser(description='Train 3-step compress')
    parser.add_argument('--input_path', '-i', required=True, type=str, help='Preprocess data path (s1, s2, s3 folders)')
    parser.add_argument('--out_path', '-o', required=True, type=str, help='Model save path')
    parser.add_argument('--log_path', '-l', default=None, type=str, help='Log directory path')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--gpus', '-d', type=str, default='0', help='GPU ID')
    parser.add_argument('--epochs', '-e', type=int, default=101, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42)
    return parser

if __name__ == '__main__':
    args = build_parser().parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # 如果没有提供 log_path，则默认在 out_path 下
    log_p = args.log_path if args.log_path else args.out_path

    main(args.input_path, args.out_path, args.gpus, args.batch_size, log_p, epochs=args.epochs, seed=args.seed)
