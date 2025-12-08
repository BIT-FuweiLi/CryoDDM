import numpy as np
import mrcfile as mf
import os
import glob

def get_noise(noise_label, in_path, target_path):
    # mrc_list = glob.glob(os.path.join(in_path,'*.mrc'))
    noise_label = np.genfromtxt(noise_label, dtype=('U40', 'i4', 'i4'), encoding='utf-8')
    for i in range(len(noise_label)):
        temp = mf.read(os.path.join(in_path, noise_label[i][0]))
        noise = temp[noise_label[i][2]-128:noise_label[i][2]+128,noise_label[i][1]-128:noise_label[i][1]+128]
        mf.write(os.path.join(target_path,str(i)+'.mrc'), noise.astype(np.float32), overwrite=True)
    return True

get_noise('temp_data/10017/10017.txt', '/data/lifuwei/SPA_data/EMPIAR-LFW/10017/normal', '/data/lifuwei/small_protein/temp_data/dual/10017/noise_patch')

