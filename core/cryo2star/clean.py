import re
import os
# 文件路径
input_file = "particles_relion.star"
output_file = "cleaned_particles_relion.star"

# 正则表达式用于识别需要处理的行
pattern = re.compile(r'\d+@J\d+/extract/.*?_|J\d+/imported/.*?_')

# 处理函数，清理文件名
def clean_filenames(line):
    # 使用正则表达式删除前缀
    line_spl = line.split(" ")
    name0 = os.path.basename(line_spl[0])
    name1 = os.path.basename(line_spl[1])
    name0 = "_".join(name0.split("_")[1:])
    name1 = "_".join(name1.split("_")[1:])
    line_spl[0] = name0
    line_spl[1] = name1 
    cleaned_line = " ".join(line_spl)
    return cleaned_line

# 打开输入和输出文件
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    data_started = False

    for line in infile:
        # 检查是否到了数据部分 (通常数据部分包含某些标记)
        if not data_started:
            if line.strip().startswith('data_'):
                data_started = True
            outfile.write(line)  # 写入头部信息
        else:
            # 如果是数据行，应用清理
            if pattern.search(line):
                cleaned_line = clean_filenames(line)
                outfile.write(cleaned_line)
            else:
                outfile.write(line)  # 不是数据行的部分原样写出

print("清理完成，结果已保存至", output_file)
