#!/usr/bin/env python3
"""
替代 relion_star_handler 的功能：
对 star 文件中的 rlnCoordinateY 列执行 -1 * value + add_value 操作

用法: python invert_coordinateY.py <input.star> <output.star> <add_value>
"""

import sys
import starfile


def process_star_file(input_file, output_file, add_value):
    # 读取 star 文件
    df = starfile.read(input_file)

    # 修改 rlnCoordinateY 列: -1 * value + add_value
    df['rlnCoordinateY'] = -1 * df['rlnCoordinateY'] + add_value

    # 写入输出文件
    starfile.write(df, output_file, overwrite=True)

    print(f"处理完成: {input_file} -> {output_file}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input.star> <output.star> <add_value>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    add_value = float(sys.argv[3])

    process_star_file(input_file, output_file, add_value)
