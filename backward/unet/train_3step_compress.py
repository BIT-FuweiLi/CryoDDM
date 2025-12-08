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
import unet2d
import sys
import util_self
import torchvision.transforms as transforms
import copy
sys.path.append("/data/lifuwei/small_protein/code/source_code4/backward/unet")
def main(preprocess_path, model_save_path, gpus, batch_size):
    temp = str(time.asctime())+'.log'
    log_file = util_self.init_log(os.path.join('/data/lifuwei/small_protein/code/source_code4/backward/unet/.log',temp))
    log_file.write('Command: python ' + ' '.join(sys.argv) + '\n')
    log_file.write('reading data ~~\n')
    
    step1_path =os.path.join(preprocess_path, 's1')
    log_file.write('reading data from '+ step1_path +'\n')
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

    torch.cuda.set_device('cuda:{}'.format(gpus))
    model = unet2d.UDenoiseNet().cuda()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    EPOCHES = 101
    train_loss = []
    s1_train_epochs_loss = []
    s2_train_epochs_loss = []
    s3_train_epochs_loss = []
    valid_epochs_loss = []
    log_file.write('training epoches ~~\n')

    # 早停机制参数
    patience = 15
    min_delta = 0.001
    best_val_loss = float('inf')
    counter = 0

    for epoch in tqdm(range(EPOCHES),file=log_file):
        model.train()
        s1_train_epoch_loss = []
        s2_train_epoch_loss = []
        s3_train_epoch_loss = []
        log_file.write('training batches ~~\n')

        for batch_idx, (inputs, targets) in enumerate(s1_train_dataloader):
            inputs = inputs.unsqueeze(1).cuda()
            targets = targets.unsqueeze(1).cuda()
            outputs = model(inputs)
            optimizer.zero_grad()
            mse_loss = criterion(outputs,targets)
            loss = mse_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            s1_train_epoch_loss.append(loss.item())
        
        for batch_idx, (inputs, targets) in enumerate(s2_train_dataloader):
            inputs = inputs.unsqueeze(1).cuda()
            targets = targets.unsqueeze(1).cuda()
            outputs = model(inputs)
            optimizer.zero_grad()
            mse_loss = criterion(outputs,targets)
            loss = mse_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            s2_train_epoch_loss.append(loss.item())

        for batch_idx, (inputs, targets) in enumerate(s3_train_dataloader):
            inputs = inputs.unsqueeze(1).cuda()
            targets = targets.unsqueeze(1).cuda()
            outputs = model(inputs)
            optimizer.zero_grad()
            mse_loss = criterion(outputs,targets)
            loss = mse_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            s3_train_epoch_loss.append(loss.item())

        if (epoch+1) % 1 == 0:
            torch.save(model,os.path.join(model_save_path, str(epoch+1)+'.pth'))
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(model_save_path,'checkpoint.pth'))

        s1_train_epochs_loss.append(np.average(s1_train_epoch_loss))
        s2_train_epochs_loss.append(np.average(s2_train_epoch_loss))
        s3_train_epochs_loss.append(np.average(s3_train_epoch_loss))
        log_file.write('epoch:'+str(epoch)+' s1_loss:'+str(s1_train_epochs_loss[-1])+' s2_loss:'+str(s2_train_epochs_loss[-1])+' s3_loss:'+str(s3_train_epochs_loss[-1])+'\n')

        if (epoch+1) % 1 == 0:
            log_file.write('valid modeling ...\n')
            model.eval()
            with torch.no_grad():
                valid_epoch_loss = []
                for idx,(inputs, labels) in enumerate(val_dataloader):
                    inputs = inputs.unsqueeze(1).cuda()
                    labels = labels.unsqueeze(1).cuda()
                    outputs = model(inputs)
                    loss = criterion(labels,outputs)
                    valid_epoch_loss.append(loss.item())
                valid_epochs_loss.append(np.average(valid_epoch_loss))
            log_file.write('epoch:'+str(epoch)+' valid_loss:'+str(valid_epochs_loss[-1])+'\n')

            # # early stop
            # if valid_epochs_loss[-1] < best_val_loss - min_delta:
            #     best_val_loss = valid_epochs_loss[-1]
            #     counter = 0
            # else:
            #     counter += 1
            #     if counter >= patience:
            #         log_file.write('Early stopping at epoch {}\n'.format(epoch))
            #         break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='org_job 2 aim_job')
    parser.add_argument('--input_path', '-i', default='/data/lifuwei/SPA_data/foward_data/10059/proright', type=str,help ='preprocess_path')
    parser.add_argument('--out_path', '-o', type=str, default = '/data/lifuwei/SPA_data/dfdn_models/3step/10059/proright',help='model_save_path')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch_size')
    parser.add_argument('--gpus', '-d', type=str,default = 1, help ='gpus')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        
    '''
    eg:
    python /storage/zhuxu/lfw/small_protien2/source_code4/backward/unet/train_3step_compress.py -i /storage/zhuxu/lfw/small_protien2/3step_data/10025/res_linear -o /storage/zhuxu/lfw/small_protien2/models/10025res_linear -b 48 -d 3
    '''
    main(args.input_path,args.out_path,args.gpus,args.batch_size)