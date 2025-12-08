#!/bin/bash
#SBATCH  -J c25s5                        ##作业名
#SBATCH  --cpus-per-task=16                  ##每个任务有几个cpu核心
#SBATCH  --ntasks-per-node=1                ##总共有几个任务
#SBATCH  -N 1                               ##需要几个计算节点
#SBATCH  -n 1                               ##我总共有多少个进程
#SBATCH  --qos=cpu96                          ##指定qos
#SBATCH  -p cpu                                ##指定队列名
#SBATCH  -o /storage/zhuxu/lfw/small_protien2/source_code3/forward/.log/run.out                          ##输出日>志
#SBATCH  -e /storage/zhuxu/lfw/small_protien2/source_code3/forward/.log/run.err                          ##错误日>志
#SBATCH --dependency=afterok:231920
cd /storage/zhuxu/lfw/small_protien2/source_code3/
python forward/forward_chunk.py -i /storage/zhuxu/lfw/small_protien2/dual_data/10025/small_compare_100/rand/ \
 -o /storage/zhuxu/lfw/small_protien2/dual_data/10025/small_compare_100/rand_s5_chunk/
