#!/bin/bash
#SBATCH  -J dndf                        ##作业名
#SBATCH  --cpus-per-task=16                  ##每个任务有几个cpu核心
#SBATCH  --ntasks-per-node=1                ##总共有几个任务
#SBATCH  -N 1                               ##需要几个计算节点
#SBATCH  -n 1                               ##我总共有多少个进程
#SBATCH  --qos=a100g1_2                          ##指定qos
#SBATCH  -p a100                                ##指定队列名
#SBATCH  -o /storage/zhuxu/lfw/small_protien2/source_code2/forward/.log/rundual.out                          ##输出日>志
#SBATCH  -e /storage/zhuxu/lfw/small_protien2/source_code2/forward/.log/rundual.err                          ##错误日>志

cd /storage/zhuxu/lfw/small_protien2/source_code2/
python forward/forward_process_dual.py -ip /storage/wanxiaohua/lfw/small_protien/diffuse_model/temp_data/10028_redo/all_normal/ -np /storage/wanxiaohua/lfw/small_protien/diffuse_model/temp_data/10028_redo/noise_patch/ \
 -op /home/zhuxu/storage/lfw/small_protien2/source_code2/dual/10028/diffusion
