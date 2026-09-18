#!/bin/bash
# Example: bash pipelineV2.sh /path/to/cryosparc/J102 /path/to/output 7424 5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
SUCCESS_COUNT=0

# 检查输入参数
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <csproject_dir> <output_dir> <y_value> [num_projects]"
    echo "  num_projects: 连续处理的项目数量 (默认: 1)"
    exit 1
fi

# 读取输入参数
INPUT_DIR1=$1
INPUT_DIR2=$2
ADD_VALUE=$3
NUM_PROJECTS=${4:-1}

if ! [[ "$NUM_PROJECTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_projects must be a positive integer, got: $NUM_PROJECTS" >&2
    exit 1
fi

# 获取输入文件夹的名称
BASE_DIR=$(basename "$INPUT_DIR1")

# 获取输入文件夹的父目录
PARENT_DIR=$(dirname "$INPUT_DIR1")

# 提取文件夹名称中的数字部分
if [[ "$BASE_DIR" =~ ^J([0-9]+)$ ]]; then
    BASE_NUM="${BASH_REMATCH[1]}"
else
    echo "Input directory name must match J<number>: $BASE_DIR" >&2
    exit 1
fi

if ! command -v csparc2star.py >/dev/null 2>&1; then
    echo "csparc2star.py was not found on PATH" >&2
    exit 1
fi

# 创建子文件夹
for i in $(seq 0 $((NUM_PROJECTS - 1))); do
    mkdir -p "$INPUT_DIR2/class$i"
done

# 生成接下来的文件夹路径
for i in $(seq 0 $((NUM_PROJECTS - 1))); do
    DIR="$PARENT_DIR/J$((BASE_NUM + i))"
    DIRS+=("$DIR")
done

# 遍历每个文件夹
for i in $(seq 0 $((NUM_PROJECTS - 1))); do
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
            FILE_NAME=$(basename "$FILE")
            if [[ "$FILE_NAME" =~ _([0-9]{3})_particles\.cs$ ]]; then
                NUM="${BASH_REMATCH[1]}"
            else
                NUM=""
            fi
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

    if (
        cd "$CLASS_DIR" || exit 1
        echo "Current directory: $(pwd)"
        echo "Running csparc2star.py with $INPUT1 and $INPUT2"
        csparc2star.py "$INPUT1" "$INPUT2" particles_relion.star || exit 1
        echo "Running clean.py"
        "$PYTHON_BIN" "$SCRIPT_DIR/clean.py" || exit 1
        echo "Deleting first 13 lines of cleaned_particles_relion.star"
        sed -i.bak '1,13d' cleaned_particles_relion.star || exit 1
        rm -f cleaned_particles_relion.star.bak
        echo "Running invert_coordinateY.py with add_to $ADD_VALUE"
        "$PYTHON_BIN" "$SCRIPT_DIR/invert_coordinateY.py" cleaned_particles_relion.star invert.star "$ADD_VALUE" || exit 1
    ); then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "Conversion failed for $DIR" >&2
    fi
done

if [ "$SUCCESS_COUNT" -ne "$NUM_PROJECTS" ]; then
    echo "Converted $SUCCESS_COUNT of $NUM_PROJECTS requested class(es); at least one conversion failed" >&2
    exit 1
fi

echo "Successfully converted $SUCCESS_COUNT class(es)"
