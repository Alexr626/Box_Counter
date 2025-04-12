# generate_shelf_filelist.py

import os

def generate_file_list(image_dir, output_txt, folder_prefix="my_images"):
    """
    image_dir: 存放所有 jpg 图片的目录，如 ./my_images
    output_txt: 生成的文件列表，比如 ./splits/shelf/train.txt
    folder_prefix: 在输出时，每一行里会拼成 folder_prefix/xxx.jpg
    """

    # 1. 收集所有jpg文件
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    # 2. 按照文件名排序
    all_files.sort()

    # 3. 逐行写入 txt
    with open(output_txt, 'w') as f:
        for i, filename in enumerate(all_files):
            # rel_path = os.path.join(folder_prefix, filename)  # 例如 "my_images/1738xxx.jpg"
            rel_path = filename
            # 在这里写入: "{相对路径} {帧索引}"
            line = f"{rel_path} {i}\n"
            f.write(line)

if __name__ == "__main__":
    # 假设：
    image_dir = "/drive2/jinboliu/csci677/images/original_images"
    output_txt = "./splits/shelf/train_files.txt"

    # 确保输出目录已存在（splits/shelf/）
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    generate_file_list(image_dir, output_txt)
    print(f"File list generated at: {output_txt}")
