import os


def generate_shelf_filelists(image_dir,
                             splits_dir="./splits/shelf",
                             folder_prefix="original",
                             train_ratio=0.8,
                             val_ratio=0.1,
                             test_ratio=0.1):
    """
    Scan all .jpg files under image_dir and sort them, and divide them into three copies of train/val/test according to the specified ratio, respectively
    Output to train_files.txt, val_files.txt, test_files.txt.
    Parameter description:
    -----------------
    image_dir: A directory that stores all jpg images, such as ./my_images
    splits_dir : The directory to output the splits file (default ./splits/shelf)
    folder_prefix: When writing text, splice the prefix before the file name to ensure that Monodepth2 can find the image according to the relative path
    train_ratio: training set proportion
    val_ratio: Verification set proportion
    test_ratio: test set proportion (train_ratio + val_ratio + test_ratio = 1)
    """

    # 1. Collect and sort all jpg
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    all_files.sort()
    num_files = len(all_files)
    if num_files == 0:
        print(f"No .jpg files found in {image_dir}. Exiting.")
        return

    # 2. Calculate partition interval
    train_end = int(num_files * train_ratio)
    val_end = int(num_files * (train_ratio + val_ratio))

    # 3. Slice index
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # 4. Make sure the output directory exists
    os.makedirs(splits_dir, exist_ok=True)

    # 5. train_files.txt, val_files.txt, test_files.txt
    train_txt = os.path.join(splits_dir, "train_files.txt")
    val_txt = os.path.join(splits_dir, "val_files.txt")
    test_txt = os.path.join(splits_dir, "test_files.txt")

    def write_list_to_file(filelist, outfile, start_idx=0):
        with open(outfile, 'w') as f:
            for i, filename in enumerate(filelist):
                # frame_index = start_idx + i
                frame_index = i
                # rel_path = os.path.join(folder_prefix, filename)
                rel_path = filename
                line = f"{rel_path} {frame_index}\n"
                f.write(line)

    write_list_to_file(train_files, train_txt)
    write_list_to_file(val_files, val_txt)
    write_list_to_file(test_files, test_txt)

    print("Generated filelists:")
    print(f"  Train: {train_txt} (count={len(train_files)})")
    print(f"  Val:   {val_txt}   (count={len(val_files)})")
    print(f"  Test:  {test_txt}  (count={len(test_files)})")
    print("Done.")


if __name__ == "__main__":

    image_dir = "/drive2/jinboliu/csci677/images/original_images"  # Directory for all JPG images
    splits_dir = "./splits/shelf"
    folder_prefix = "original_images"
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
