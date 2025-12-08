import numpy as np
from tqdm import *
import mrcfile as mf 
import glob
import os
import argparse
import time

def normalize_save(org_path, save_path):
    mrc_list = sorted(glob.glob(os.path.join(org_path,"*.mrc")))

    for i in trange(len(mrc_list)):
        try:
            temp = mf.read(mrc_list[i])
            normal = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
            mf.write(os.path.join(save_path, os.path.basename(mrc_list[i])), normal.astype(np.float32))
        except Exception as e:
            print(f"Error processing file {mrc_list[i]}: {e}")


if __name__ == '__main__':
    '''
    input should be single frame mrc folder
    output should be an existing folder
    example: python noramlize.py  -i /data/10017/ -o temp_data/10017/normal
    '''
    parser = argparse.ArgumentParser(description='normalize mrc data')
    parser.add_argument('--input_path', '-i', type=str,help ='org_data_path', default= ' ')
    parser.add_argument('--output_path', '-o', type=str,help='normal_data_path', default= ' ')
    args = parser.parse_args()
    # print(args)
    # time.sleep(1800)
    normalize_save(args.input_path, args.output_path)

