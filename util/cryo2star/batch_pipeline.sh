#!/bin/bash
# eg:bash /data/lifuwei/small_protein/code/source_code4/util/cryo2star/batch_pipeline.sh /data/lifuwei/cryosparc_data/CS-10025res-linear-cryoddm/J102 /data/lifuwei/relion_projects/cryosparc_data/10025res_linear/onlys1 7424
# 检查输入参数
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_dir1> <input_dir2> <add_value>"
    exit 1
fi

# 读取输入参数
INPUT_DIR1=$1
INPUT_DIR2=$2
ADD_VALUE=$3

# 获取输入文件夹的名称
BASE_DIR=$(basename "$INPUT_DIR1")

# 获取输入文件夹的父目录
PARENT_DIR=$(dirname "$INPUT_DIR1")

# 提取文件夹名称中的数字部分
BASE_NUM=$(echo "$BASE_DIR" | grep -oP '(?<=J)\d+')

# 创建子文件夹
for i in {0..9}; do
    mkdir -p "$INPUT_DIR2/class$i"
done

# 生成接下来的五个文件夹路径
for i in {0..9}; do
    DIR="$PARENT_DIR/J$((BASE_NUM + i))"
    DIRS+=("$DIR")
done

# 遍历每个文件夹
for i in {0..9}; do
    DIR="${DIRS[$i]}"

    # 检查文件夹是否存在
    if [ ! -d "$DIR" ]; then
        echo "Directory $DIR does not exist"
        continue
    fi

    # 获取文件夹名称
    DIR_NAME=$(basename "$DIR")

    # 找到文件夹中数字最大的 _particles.cs 文件
    MAX_PARTICLES_FILE=""
    MAX_PARTICLES_NUM=-1

    for FILE in "$DIR"/*_particles.cs; do
        if [ -f "$FILE" ]; then  # 确保是文件
            NUM=$(echo "$FILE" | grep -oP '(?<=_)\d{3}(?=_particles.cs)')
            if [ -n "$NUM" ] && [ "$NUM" -gt "$MAX_PARTICLES_NUM" ]; then
                MAX_PARTICLES_NUM=$NUM
                MAX_PARTICLES_FILE="$FILE"
            fi
        fi
    done

    if [ -z "$MAX_PARTICLES_FILE" ]; then
        echo "No _particles.cs files found in directory $DIR"
        continue
    fi
    INPUT1="$MAX_PARTICLES_FILE"
    INPUT2="$DIR/${DIR_NAME}_passthrough_particles.cs"

    # 检查文件是否存在
    if [ ! -f "$INPUT1" ] || [ ! -f "$INPUT2" ]; then
        echo "Files $INPUT1 or $INPUT2 not found in directory $DIR"
        continue
    fi

    CLASS_DIR="$INPUT_DIR2/class$i"

    # 保存当前目录
    ORIGINAL_DIR=$(pwd)

    # 进入 class 文件夹
    cd "$CLASS_DIR" || { echo "Failed to change directory to $CLASS_DIR"; continue; }
    echo "Current directory: $(pwd)"

    # 运行 csparc2star.py 脚本
    echo "Running csparc2star.py with $INPUT1 and $INPUT2"
    csparc2star.py "$INPUT1" "$INPUT2" particles_relion.star 2>/dev/null

    # 运行 trans.py 脚本
    echo "Running clean.py"
    python "/data/lifuwei/small_protein/code/source_code4/util/cryo2star/clean.py"

    # 删除 cleaned_particles_relion.star 文件的前 13 行
    echo "Deleting first 13 lines of cleaned_particles_relion.star"
    sed -i '1,13d' cleaned_particles_relion.star

    # 运行 relion_star_handler 并修改 rlnCoordinateY 列
    echo "Running relion_star_handler with add_to $ADD_VALUE"
    relion_star_handler --i cleaned_particles_relion.star --o invert.star --operate rlnCoordinateY --multiply_by -1 --add_to "$ADD_VALUE" 2>/dev/null

    # 返回原始目录
    cd "$ORIGINAL_DIR" || { echo "Failed to return to $ORIGINAL_DIR"; exit 1; }
done
