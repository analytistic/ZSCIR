#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 定义输出目录
OUTPUT_DIR="images"

# 遍历所有 asin2url 开头的 txt 文件
for file in asin2url.dress.txt.failed.txt.failed.txt; do
    if [ -f "$file" ]; then
        echo "========================================"
        echo "Processing file: $file"
        echo "========================================"
        
        # 调用 Python 脚本
        # $file 是输入文件
        # --output_dir 指定保存路径
        python3 downpipline.py "$file" --output_dir "$OUTPUT_DIR"
        
        echo "Done with $file"
        echo ""
    fi
done

echo "All downloads completed."
