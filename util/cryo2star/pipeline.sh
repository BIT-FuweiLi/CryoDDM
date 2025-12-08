#!/bin/bash

# 检查输入参数
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input1> <input2> <add_value>"
    exit 1
fi

# 读取输入参数
INPUT1=$1
INPUT2=$2
ADD_VALUE=$3

# 运行 csparc2star.py 脚本
csparc2star.py "$INPUT1" "$INPUT2" particles_relion.star

# 运行 trans.py 脚本
python "/data/lifuwei/small_protein/code/source_code4/util/cryo2star/clean.py"

# 删除 cleaned_particles_relion.star 文件的前 13 行
sed -i '1,13d' cleaned_particles_relion.star

# 运行 relion_star_handler 并修改 rlnCoordinateY 列
relion_star_handler --i cleaned_particles_relion.star --o invert.star --operate rlnCoordinateY --multiply_by -1 --add_to "$ADD_VALUE"

#eg bash /home/lifuwei/lifuwei/small_protein/code/source_code4/util/cryo2star/pipeline.sh   /data/lifuwei/cryosparc_data/CS-10005/J90/J90_passthrough_particles.cs 7424