import os


def generate_shelf_filelists(image_dir,
                             splits_dir="./splits/shelf",
                             folder_prefix="my_images",
                             train_ratio=0.8,
                             val_ratio=0.1,
                             test_ratio=0.1):
    """
    扫描 image_dir 下的所有 .jpg 文件并排序，按指定比例切分为 train/val/test 三份，分别
    输出到 train_files.txt, val_files.txt, test_files.txt.

    参数说明：
    -----------
    image_dir    : 存放所有 jpg 图片的目录，如 ./my_images
    splits_dir   : 要输出 splits 文件的目录（默认 ./splits/shelf）
    folder_prefix: 写入文本时，拼接到文件名之前的前缀，保证 Monodepth2 能按相对路径找到图像
    train_ratio  : 训练集占比
    val_ratio    : 验证集占比
    test_ratio   : 测试集占比（train_ratio + val_ratio + test_ratio = 1）
    """

    # 1. 收集并排序所有 jpg
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    all_files.sort()
    num_files = len(all_files)
    if num_files == 0:
        print(f"No .jpg files found in {image_dir}. Exiting.")
        return

    # 2. 计算切分区间
    train_end = int(num_files * train_ratio)
    val_end = int(num_files * (train_ratio + val_ratio))
    # 剩下的就是 test

    # 3. 切分索引
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # 4. 确保输出目录存在
    os.makedirs(splits_dir, exist_ok=True)

    # 5. 依次写 train_files.txt, val_files.txt, test_files.txt
    train_txt = os.path.join(splits_dir, "train_files.txt")
    val_txt = os.path.join(splits_dir, "val_files.txt")
    test_txt = os.path.join(splits_dir, "test_files.txt")

    # 写入函数
    def write_list_to_file(filelist, outfile, start_idx=0):
        with open(outfile, 'w') as f:
            for i, filename in enumerate(filelist):
                # frame_index = start_idx + i
                # 这里如果希望全局连续编号，就可以加上 start_idx，暂时我们就从 0 开始算
                frame_index = i
                # rel_path = os.path.join(folder_prefix, filename)
                rel_path = filename
                line = f"{rel_path} {frame_index}\n"
                f.write(line)

    # 写三个文件
    write_list_to_file(train_files, train_txt)
    write_list_to_file(val_files, val_txt)
    write_list_to_file(test_files, test_txt)

    print("Generated filelists:")
    print(f"  Train: {train_txt} (count={len(train_files)})")
    print(f"  Val:   {val_txt}   (count={len(val_files)})")
    print(f"  Test:  {test_txt}  (count={len(test_files)})")
    print("Done.")


if __name__ == "__main__":
    # 你可以根据实际情况修改这些变量
    image_dir = "/drive2/jinboliu/csci677/images/original_images"  # 存放所有 JPG 图片的目录
    splits_dir = "./splits/shelf"  # 输出 splits 文件的目录
    folder_prefix = "my_images"  # 文本里写的相对路径前缀
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    generate_shelf_filelists(
        image_dir=image_dir,
        splits_dir=splits_dir,
        folder_prefix=folder_prefix,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
