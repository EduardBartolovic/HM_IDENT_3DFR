import os
import shutil

from tqdm import tqdm


def copy_filtered_dataset(src_dir, dest_dir, filters):
    """
    Copies images from src_dir to dest_dir while preserving the structure,
    but only includes images whose names contain at least one of the filter strings.

    :param src_dir: Path to the source dataset directory.
    :param dest_dir: Path to the destination dataset directory.
    :param filters: List of substrings that should be contained in image filenames.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for split in ['train', 'validation']:
        split_src = os.path.join(src_dir, split)
        split_dest = os.path.join(dest_dir, split)

        if not os.path.exists(split_src):
            continue

        for identity in tqdm(os.listdir(split_src)):
            identity_src = os.path.join(split_src, identity)
            identity_dest = os.path.join(split_dest, identity)

            if not os.path.isdir(identity_src):
                continue

            os.makedirs(identity_dest, exist_ok=True)

            for img in os.listdir(identity_src):
                if img[40:-10] in filters:
                    shutil.copy(os.path.join(identity_src, img), os.path.join(identity_dest, img))

    print(f"Filtered dataset copied to {dest_dir}")


if __name__ == '__main__':

    src_dataset = "F:\\Face\\data\\datasets8\\test_vox2train"
    dest_dataset = "F:\\Face\\data\\datasets8\\test_vox2train"

    filters = ["0_0", "-25_-25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)

    exit()

    src_dataset = "F:\\Face\\data\\datasets8\\test_rgb_bff"
    dest_dataset = "F:\\Face\\data\\datasets8\\test_rgb_bff"

    filters = ["-25_-25", "-25_0", "-25_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)
    filters = ["25_-25", "25_0", "25_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)

    filters = ["0_0", "-25_-25", "25_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)
    filters = ["0_0", "25_-25", "-25_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)

    filters = ["0_0", "-10_-10", "10_10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)
    filters = ["0_0", "10_-10", "-10_10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)

    filters = ["0_0", "0_-25", "0_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)
    filters = ["0_0", "25_0", "25_0"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)

    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)
    filters = ["0_0", "0_-10", "0_10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)
    filters = ["0_0", "10_0", "10_0"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[0]+"_"+filters[1]+"_"+filters[2], filters)

    exit()


    filters = ["0_0", "-25_-25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "-25_-10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "-25_0"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "-25_10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "-25_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)

    filters = ["0_0", "-10_-25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "-10_-10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "-10_0"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "-10_10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "-10_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)

    filters = ["0_0", "0_-25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "0_-10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    #filters = ["0_0", ]
    #copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "0_10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "0_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)

    filters = ["0_0", "25_-25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "25_-10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "25_0"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "25_10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "25_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)

    filters = ["0_0", "10_-25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "10_-10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "10_0"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "10_10"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
    filters = ["0_0", "10_25"]
    copy_filtered_dataset(src_dataset, dest_dataset+filters[1], filters)
