import os
from tqdm import *
import torch
import numpy as np
import torch.utils.data as data
import mrcfile as mf
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import json
from tqdm import *
import math
import glob
import copy
def init_log(log_path = '3Dunet_n2n/.log/train_datasets_mini.log'):
    if os.path.exists(log_path):
        log_file = open(log_path, "a")
    elif os.path.exists(os.path.dirname(log_path)):
        log_file = open(log_path, "w")
    else:
        os.makedirs(os.path.dirname(log_path))
        log_file = open(log_path, "w")
    return log_file

def get_rawdata_list(path, log_file):
    # 检测路径是否合理
    if not os.path.exists(path):
        log_file.write('mrc file path is not exist\n')
    # 获取所有.mrc文件
    file_list = os.listdir(path)
    mrc_list = []
    for file in file_list:
        if file.endswith('.mrc'):
            mrc_list.append(file)
    mrc_list.sort()
    # mrc_data_np = []

    # for i in tqdm(range(len(mrc_list)), file=log_file):
    #     log_file.write('reading {} ~~\n'.format(mrc_list[i]))
    #     mrc_data_np.append(np.array(mf.read(path+'/'+mrc_list[i])))
    # # mrc_data_np = np.array(mrc_data_np)

    return mrc_list

def get_rawdata(path, log_file):
    # 检测路径是否合理
    if not os.path.exists(path):
        log_file.write('mrc file path is not exist\n')
    # 获取所有.mrc文件
    file_list = os.listdir(path)
    mrc_list = []
    for file in file_list:
        if file.endswith('.mrc'):
            mrc_list.append(file)
    mrc_list.sort()
    mrc_data_np = []

    for i in tqdm(range(len(mrc_list)), file=log_file):
        log_file.write('reading {} ~~\n'.format(mrc_list[i]))
        mrc_data_np.append(np.array(mf.read(path+'/'+mrc_list[i])))
    # mrc_data_np = np.array(mrc_data_np)

    return mrc_data_np, mrc_list


def crop_data(mrc_data, shape, padding, log_file, cval):
    len_y = math.ceil(mrc_data.shape[0]/shape[0])
    len_x = math.ceil(mrc_data.shape[1]/shape[1])
    subsets =  np.zeros((len_x*len_y, shape[0]+2*padding, shape[1]+2*padding))
    subsets.fill(cval)
    matchs = []
    subsets_index = 0
    # for j in tqdm(range(len_y),file=log_file):
    for j in range(len_y):
        for k in range(len_x):
            subset = np.zeros((shape[0]+2*padding, shape[1]+2*padding))
            subset.fill(cval)
            sj = max(0, j*shape[0]-padding)
            ej = min(mrc_data.shape[0], (j+1)*shape[0]+padding)
            sk = max(0, k*shape[1]-padding)
            ek = min(mrc_data.shape[1], (k+1)*shape[1]+padding)


            sjc = padding - j*shape[0] + sj
            ejc = sjc + (ej - sj)
            skc = padding - k*shape[1] + sk
            ekc = skc + (ek - sk)

            subset[sjc:ejc,skc:ekc] = mrc_data[sj:ej,sk:ek]
            

            # subsets.append(mrc_data[i*shape[0]:(i+1)*shape[0],j*shape[1]:(j+1)*shape[1],k*shape[2]:(k+1)*shape[2]])
            subsets[subsets_index] = subset
            subsets_index += 1
            matchs.append([sj,ej,sk,ek,sjc,ejc,skc,ekc])
    return subsets, (len_y,len_x), matchs 

def concat_data(subsets, len_yx, match, log_file, padding):
    shape_y = subsets[0].shape[0]
    shape_x = subsets[0].shape[1]
    data_2d = np.zeros((len_yx[0]*(shape_y-2*padding),len_yx[1]*(shape_x-2*padding)))
    # index = 0
    # for i in tqdm(range(len(match)),file=log_file):
    for i in range(len(match)):
        sj, ej, sk, ek, sjc, ejc, skc, ekc = match[i]

        if sj != 0:
            sj = sj + padding
        if sk != 0:
            sk = sk + padding
        ej = sj + shape_y-2*padding
        ek = sk + shape_x-2*padding

        # if ei <= data_3d.shape[0]-shape_z:
        #     ei = ei - padding
        # if ej <= data_3d.shape[1]-shape_y:
        #     ej = ej - padding
        # if ek <= data_3d.shape[2]-shape_x:
        #     ek = ek - padding
        # # ei = ei-padding
        # ej = ej-padding
        # ek = ek-padding

        # sic = sic+padding
        # sjc = sjc+padding
        # skc = skc+padding
        # eic = eic-padding
        # ejc = ejc-padding
        # ekc = ekc-padding

        data_2d[sj:ej,sk:ek] = subsets[i][padding:-padding,padding:-padding]
    # for i in tqdm(range(len_zyx[0]),file=log_file):
    #     for j in range(len_zyx[1]):
    #         for k in range(len_zyx[2]):
    #             data_3d[] = subsets[index]
    #             index += 1
    return data_2d

#使用实例
# log_file.write('concating data ~~')
# con_test = concat_data(crop_test, split)
# mf.write('crop_test.mrc', con_test.astype(np.float32), overwrite=True)

def crop_dataset(dataset, shape, log_file, padding=64, cval = 0):
    subsets = []
    sizes = []
    matches = []
    log_file.write('croping datasets ~~\n')
    for i in tqdm(range(len(dataset)),file=log_file):
        log_file.write('croping dataset {} ~~\n'.format(i))
        subset, size, match = crop_data(dataset[i], shape, padding,log_file, cval)
        if i == 0:
            subsets = subset   
        else:
            subsets = np.concatenate((subsets, subset), axis=0)
        sizes.append(size)
        matches.append(match)
    return subsets, sizes, matches

def concat_dataset(subsets, sizes, matches, log_file, padding):
    log_file.write('concating datasets ~~\n')
    datasets=[]
    start = 0
    end = 0
    for i in tqdm(range(len(sizes)),file=log_file):
        log_file.write('concating dataset {}\n'.format(i))
        # end += sizes[i][0]*sizes[i][1]*sizes[i][2]
        end += len(matches[i])
        data_3d = concat_data(subsets[start:end], sizes[i], matches[i], log_file, padding)
        start = end
        datasets.append(data_3d)
    
    return datasets


# class PatchDataset:
#     def __init__(self, tomo, patch_size=96, padding=48):
#         self.tomo = tomo
#         self.patch_size = patch_size
#         self.padding = padding

#         nz,ny,nx = tomo.shape

#         pz = int(np.ceil(nz/patch_size))
#         py = int(np.ceil(ny/patch_size))
#         px = int(np.ceil(nx/patch_size))
#         self.shape = (pz,py,px)
#         self.num_patches = pz*py*px


#     def __len__(self):
#         return self.num_patches

#     def __getitem__(self, patch):
#         # patch index
#         i,j,k = np.unravel_index(patch, self.shape)

#         patch_size = self.patch_size
#         padding = self.padding
#         tomo = self.tomo

#         # pixel index
#         i = patch_size*i
#         j = patch_size*j
#         k = patch_size*k

#         # make padded patch
#         d = patch_size + 2*padding
#         x = np.zeros((d, d, d), dtype=np.float32)

#         # index in tomogram
#         si = max(0, i-padding)
#         ei = min(tomo.shape[0], i+patch_size+padding)
#         sj = max(0, j-padding)
#         ej = min(tomo.shape[1], j+patch_size+padding)
#         sk = max(0, k-padding)
#         ek = min(tomo.shape[2], k+patch_size+padding)

#         # index in crop
#         sic = padding - i + si
#         eic = sic + (ei - si)
#         sjc = padding - j + sj
#         ejc = sjc + (ej - sj)
#         skc = padding - k + sk
#         ekc = skc + (ek - sk)

#         x[sic:eic,sjc:ejc,skc:ekc] = tomo[si:ei,sj:ej,sk:ek]
#         return np.array((i,j,k), dtype=int),x
# mydata = PatchDataset(train_input_mrc_raw_data[0], patch_size=128, padding=128)
# batch_iterator = torch.utils.data.DataLoader(mydata, batch_size=16, shuffle=False)
# for index,x in batch_iterator:
#     print(index)
#     print(x)
class CustomDataset(data.Dataset):
   def __init__(self, inputs, labels):
       self.inputs = inputs
       self.labels = labels

   def __len__(self):
       return len(self.inputs)

   def __getitem__(self, index):
       inputs = torch.tensor(self.inputs[index], dtype=torch.float32)
       labels = torch.tensor(self.labels[index], dtype=torch.float32)
       return inputs, labels

def min_max_normalize_3d_data(data, min_val='',range_val=''):
    if min_val == '' and range_val == '':
        # 获取最大和最小值
        max_val = np.max(data)
        min_val = np.min(data)

        # 计算归一化范围
        range_val = max_val - min_val

    # 对数据进行归一化
    normalized_data = (data - min_val) / range_val
    
    return normalized_data, min_val, range_val

def min_max_denormalize_3d_data(normalized_data,min_val,range_val):
    return normalized_data *range_val + min_val

def normalize_3d_data(data,mean='',std=''):
   """
   对三维数据进行均值归一化
   :param data: 三维数据,形状为 (N, 3)
   :return: 均值归一化后的数据
   """
   # 如果有mean，std则使用，否则不使用
   if mean != '' and std != '':
        normalized_data = (data - mean) / std
   else:
        # 计算每个分量的均值
        mean = np.mean(data)
        
        # 计算每个分量的方差
        variance = np.var(data)
        
        # 计算每个分量的标准差
        std = np.sqrt(variance)
        
        # 进行均值归一化
        normalized_data = (data - mean) / std
   
   return normalized_data, mean, std

def denormalize_3d_data(normalized_data, mean, std):
   """
   根据均值和标准差反归一化三维数据
   :param normalized_data: 归一化后的三维数据,形状为 (N, 3)
   :param mean: 每个分量的均值,形状为 (3,)
   :param std_dev: 每个分量的标准差,形状为 (3,)
   :return: 反归一化后的三维数据
   """
   # 计算每个分量的反标准化值
   reverse_standardization_values = (normalized_data * std) + mean
   
   # 进行反归一化
   denormalized_data = reverse_standardization_values
   
   return denormalized_data
# 使用例子
# inputMrc,mean,std = normalize_3d_data(inputMrc)
# labelMrc,_,_ = normalize_3d_data(labelMrc,mean,std)

# SSIM_Loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    _3D_window = _2D_window.mm(_2D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_loss(img1, img2, window_size = 11, size_average = True):
    (_, _,channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def load_datacrops(path,jump_list,log_file):
    log_file.write('load_datacrops\n')
    if not os.path.exists(path):
        log_file.write('path not exists!\n')
        exit()
    
    josn_path = os.path.join(path, 'data_info.json')
    f = open(josn_path, 'r')
    content = f.read()
    data_info = json.loads(content)
    f.close()

    if jump_list[0] == 1:
        log_file.write('jumping train_dataset \n')
        train_input, train_label = 0, 0
    else:
        train_input_path = os.path.join(path,'train/input')
        train_list = glob.glob(os.path.join(train_input_path,"*.mrcs"))
        train_input = []
        log_file.write('load_input_datacrops\n')
        for i in tqdm(range(len(train_list)),file=log_file):
            train_input.append(mf.read(train_list[i]))
        train_input = np.array(train_input)

        train_label_path = os.path.join(path,'train/label')
        train_label_list = glob.glob(os.path.join(train_label_path, '*.mrcs'))
        train_label = []
        log_file.write('load_label_datacrops\n')
        for i in tqdm(range(len(train_label_list)),file=log_file):
            train_label.append(mf.read(train_label_list[i]))
        train_label = np.array(train_label)
        train_input = train_input.reshape(-1,train_input.shape[-2],train_input.shape[-1])
        train_label = train_label.reshape(-1,train_label.shape[-2],train_label.shape[-1])
    
    if jump_list[1] == 1:
        log_file.write('jumping val_dataset!\n')
        val_input, val_label = 0, 0
    else:
        val_input_path = os.path.join(path,'val/input')
        val_list = glob.glob(os.path.join(val_input_path,"*.mrcs"))
        val_input = []
        for i in range(len(val_list)):
            val_input.append(mf.read(val_list[i]))
        val_input = np.array(val_input)
        val_label_path = os.path.join(path,'val/label')
        val_label_list = glob.glob(os.path.join(val_label_path, '*.mrcs'))
        val_label = []
        for i in range(len(val_label_list)):
            val_label.append(mf.read(val_label_list[i]))
        val_label = np.array(val_label)
        val_input = val_input.reshape(-1, val_input.shape[-2], val_input.shape[-1])
        val_label = val_label.reshape(-1, val_label.shape[-2], val_label.shape[-1])


    if jump_list[2]==1:
        log_file.write('jumping test_dataset!\n')
        test_input, test_label = 0,0
    else:
        test_input_path = os.path.join(path,'test/input')
        test_list = glob.glob(os.path.join(test_input_path,"*.mrcs"))

        test_input = []
        for i in range(len(test_list)):
            test_input.append(mf.read(test_list[i]))
        test_input = np.array(test_input)
        test_label_path = os.path.join(path,'test/label')
        test_label_list = glob.glob(os.path.join(test_label_path, '*.mrcs'))
        test_label = []

        for i in range(len(test_label_list)):
            test_label.append(mf.read(test_label_list[i]))
        test_label = np.array(test_label)
        test_input = test_input.reshape(-1, test_input.shape[-2],test_input.shape[-1])
        test_label = test_label.reshape(-1, test_label.shape[-2], test_label.shape[-1])

    crop_shape = data_info['crop_shape']
    padding = data_info['padding']
    
    return train_input, train_label, val_input, val_label, test_input, test_label, data_info

def load_data_compress(path,extra_path,log_file):
    train_data_path = os.path.join(path,extra_path)
    log_file.write('loading data from:'+str(train_data_path))
    train_list = glob.glob(os.path.join(train_data_path,"*.mrcs"))
    tempdata = mf.read(train_list[0])
    train_data = np.zeros((len(train_list),tempdata.shape[0],tempdata.shape[1],tempdata.shape[2]),dtype=np.float32)
    log_file.write('load_input_datacrops\n')
    for i in tqdm(range(len(train_list)),file=log_file):
        train_data[i] = mf.read(train_list[i])
    train_data = np.array(train_data)
    train_input = copy.deepcopy(train_data[1:])
    train_label = copy.deepcopy(train_data[:-1])
    train_input = train_input.reshape(-1,train_input.shape[-2],train_input.shape[-1])
    train_label = train_label.reshape(-1,train_label.shape[-2],train_label.shape[-1])
    return train_input, train_label

def load_datacrops_compress(path,jump_list,log_file):

    log_file.write('load_datacrops\n')
    if not os.path.exists(path):
        log_file.write('path not exists!\n')
        exit()

    josn_path = os.path.join(path, 'data_info.json')
    f = open(josn_path, 'r')
    content = f.read()
    data_info = json.loads(content)
    f.close()

    if jump_list[0] == 1:
        log_file.write('jumping train_dataset \n')
        train_input, train_label = 0, 0
    else:
        train_input, train_label = load_data_compress(path,'train',log_file)
    
    if jump_list[1] == 1:
        log_file.write('jumping val_dataset!\n')
        val_input, val_label = 0, 0
    else:
        val_input, val_label = load_data_compress(path,'val',log_file)

    if jump_list[2]==1:
        log_file.write('jumping test_dataset!\n')
        test_input, test_label = 0,0
    else:
        val_input, val_label = load_data_compress(path,'test',log_file)

    return train_input, train_label, val_input, val_label, test_input, test_label, data_info

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            # if img1.is_cuda:
            #     window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return 1 - _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def mse(imageA, imageB):
    # 计算均方误差
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def psnr(imageA, imageB):
    # 计算PSNR值
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return 100  # 如果两幅图像完全相同，则返回一个很大的数表示无穷大的PSNR值
    MAX_I = 1
    return 20 * np.log10(MAX_I / np.sqrt(mse_value))









# 这一段是将高低频信息截取，还没实现
# def get_low_high_f(mrc, low_pass, r):

 
# def gaussian_filter_high_f(fshift, D):
#     # 获取索引矩阵及中心点坐标
#     h, w = fshift.shape
#     x, y = np.mgrid[0:h, 0:w]
#     center = (int((h - 1) / 2), int((w - 1) / 2))
 
#     # 计算中心距离矩阵
#     dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
 
#     # 计算变换矩阵
#     template = np.exp(- dis_square / (2 * D ** 2))
 
#     return template * fshift
 
# def gaussian_filter_low_f(fshift, D):
#     # 获取索引矩阵及中心点坐标
#     h, w = fshift.shape
#     x, y = np.mgrid[0:h, 0:w]
#     center = (int((h - 1) / 2), int((w - 1) / 2))
 
#     # 计算中心距离矩阵
#     dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
 
#     # 计算变换矩阵
#     template = 1 - np.exp(- dis_square / (2 * D ** 2)) # 高斯过滤器
 
#     return template * fshift
 
# def circle_filter_high_f(fshift, radius_ratio):
#     """
#     过滤掉除了中心区域外的高频信息
#     """
#     # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
#     template = np.zeros(fshift.shape, np.uint8)
#     crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
#     radius = int(radius_ratio * img.shape[0] / 2)
#     if len(img.shape) == 3:
#         cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
#     else:
#         cv2.circle(template, (crow, ccol), radius, 1, -1)
#     # 2, 过滤掉除了中心区域外的高频信息
#     return template * fshift
 
 
# def circle_filter_low_f(fshift, radius_ratio):
#     """
#     去除中心区域低频信息
#     """
#     # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
#     filter_img = np.ones(fshift.shape, np.uint8)
#     crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
#     radius = int(radius_ratio * img.shape[0] / 2)
#     if len(img.shape) == 3:
#         cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
#     else:
#         cv2.circle(filter_img, (crow, col), radius, 0, -1)
#     # 2 过滤中心低频部分的信息
#     return filter_img * fshift
 
 
# def ifft(fshift):
#     """
#     傅里叶逆变换
#     """
#     ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
#     iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
#     iimg = np.abs(iimg)  # 返回复数的模
#     return iimg
 
# def get_low_high_f(img, radius_ratio, D):
#     """
#     获取低频和高频部分图像
#     """
#     # 傅里叶变换
#     # np.fft.fftn
#     f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
#     fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频
 
#     # 获取低频和高频部分
#     hight_parts_fshift = circle_filter_low_f(fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
#     low_parts_fshift = circle_filter_high_f(fshift.copy(), radius_ratio=radius_ratio)
#     hight_parts_fshift =  gaussian_filter_low_f(fshift.copy(), D=D)
#     low_parts_fshift = gaussian_filter_high_f(fshift.copy(), D=D)
 
#     low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
#     high_parts_img = ifft(hight_parts_fshift)
 
#     # 显示原始图像和高通滤波处理图像
#     img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
#     img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
#                 np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)
 
#     # uint8
#     img_new_low = np.array(img_new_low * 255, np.uint8)
#     img_new_high = np.array(img_new_high * 255, np.uint8)
#     return img_new_low, img_new_high
 
 
# # 频域中使用高斯滤波器能更好的减少振铃效应
# if __name__ == '__main__':
#     radius_ratio = 0.5  # 圆形过滤器的半径：ratio * w/2
#     D = 50              # 高斯过滤器的截止频率：2 5 10 20 50 ，越小越模糊信息越少
#     img = mf.imread('butterfly2.png', cv2.IMREAD_GRAYSCALE)
#     low_freq_part_img, high_freq_part_img = get_low_high_f(img, radius_ratio=radius_ratio, D=D)  # multi channel or single
 
#     plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
#     plt.axis('off')
#     plt.subplot(132), plt.imshow(low_freq_part_img, 'gray'), plt.title('low_freq_img')
#     plt.axis('off')
#     plt.subplot(133), plt.imshow(high_freq_part_img, 'gray'), plt.title('high_freq_img')
#     plt.axis('off')
#     plt.show()

def get_mrcsdata_list(path, log_file):
    # 检测路径是否合理
    if not os.path.exists(path):
        log_file.write('mrcs file path is not exist\n')
        return []
    
    # 获取所有.mrcs文件
    file_list = os.listdir(path)
    mrcs_list = []
    for file in file_list:
        if file.endswith('.mrcs'):
            mrcs_list.append(file)
    mrcs_list.sort()
    
    return mrcs_list