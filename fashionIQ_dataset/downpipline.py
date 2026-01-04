import os
import requests
import time
import argparse
from tqdm import tqdm
import concurrent.futures

def download_image(img_id, img_url, save_dir):
    """下载单张图片的函数"""
    try:
        # 发送请求
        response = requests.get(img_url, timeout=10)
        response.raise_for_status() # 检查请求是否成功

        # 构造文件名 (使用 ID 作为文件名，后缀名为 .png)
        file_path = os.path.join(save_dir, f"{img_id}.png")

        # 如果文件已存在，跳过 (可选)
        if os.path.exists(file_path):
             return None

        # 写入文件
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"[失败] ID: {img_id} 下载出错: {e}")
        return f"{img_id}\t{img_url}"

def main():
    # 设置参数解析
    parser = argparse.ArgumentParser(description="Download FashionIQ images from a URL list file.")
    parser.add_argument("input_file", help="Path to the text file containing ID and URL pairs (e.g., asin2url.dress.txt)")
    parser.add_argument("--output_dir", default="images", help="Directory to save downloaded images")
    parser.add_argument("--workers", type=int, default=8, help="Number of threads to use")
    
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    # 创建保存目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print(f"Starting download from: {args.input_file}")
    print(f"Saving to: {args.output_dir}")

    with open(args.input_file, 'r') as f:
        raw_data = f.read()
        
    # 去除首尾空白并按行分割
    lines = raw_data.strip().split('\n')
    
    # 预处理任务列表
    tasks = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            tasks.append((parts[0].strip(), parts[1].strip()))
            
    print(f"Total {len(tasks)} images to download with {args.workers} threads.")

    failed_downloads = []

    # 使用线程池并发下载
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        futures = [executor.submit(download_image, item_id, url, args.output_dir) for item_id, url in tasks]
        
        # 使用 tqdm 显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading"):
            result = future.result()
            if result:
                failed_downloads.append(result)

    # 保存失败的记录
    if failed_downloads:
        error_log_path = args.input_file + ".failed.txt"
        print(f"\n[Warning] {len(failed_downloads)} downloads failed.")
        print(f"Saving failed URLs to: {error_log_path}")
        with open(error_log_path, 'w') as f:
            for line in failed_downloads:
                f.write(line + '\n')
    else:
        print("\nAll downloads completed successfully!")

    print(f"Finished processing {args.input_file}")

if __name__ == "__main__":
    main()