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
    line_spl = line.split()
    if len(line_spl) < 2:
        raise ValueError(f"Expected at least two STAR columns, got: {line.rstrip()}")
    name0 = os.path.basename(line_spl[0])
    name1 = os.path.basename(line_spl[1])
    name0 = "_".join(name0.split("_")[1:])
    name1 = "_".join(name1.split("_")[1:])
    line_spl[0] = name0
    line_spl[1] = name1 
    cleaned_line = " ".join(line_spl) + "\n"
    return cleaned_line

def clean_star_file(source=input_file, destination=output_file):
    with open(source, 'r') as infile, open(destination, 'w') as outfile:
        data_started = False
        for line in infile:
            if not data_started:
                if line.strip().startswith('data_'):
                    data_started = True
                outfile.write(line)
            elif pattern.search(line):
                outfile.write(clean_filenames(line))
            else:
                outfile.write(line)


if __name__ == '__main__':
    clean_star_file()
    print("清理完成，结果已保存至", output_file)
