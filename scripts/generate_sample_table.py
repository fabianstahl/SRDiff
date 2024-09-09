import os
import random
import tqdm

# Local
#inp_dir     = "~/code/DOPS-Data/s2_tiles_all/"
#gt_dir      = "~/code/DOPS-Data/data_unique/"

# Black Donkey
inp_dir     = "/sr_data/s2_tiles_all/"
gt_dir      = "/sr_data/dop20_tiles/"

partition_path = "earth_data_sample_table.txt"

inp_paths   = sorted(os.listdir(inp_dir))
gt_paths    = sorted(os.listdir(gt_dir))

def generate_labels(no_files, train_ratio, val_ratio, test_ratio, seed=42):
    random.seed(seed)

    # Calculate the number of images to assign to each label
    num_label_0 = int(train_ratio * no_files)
    num_label_1 = int(val_ratio * no_files)
    num_label_2 = no_files - num_label_0 - num_label_1  # Remaining images go to label 2

    # Create a list of labels with the desired ratios and shuffle them
    labels      = [0] * num_label_0 + [1] * num_label_1 + [2] * num_label_2
    random.shuffle(labels)
    return labels



with open(partition_path, "w") as f:

    assert len(inp_paths) == len(gt_paths)

    labels = generate_labels(len(inp_paths), 0.8, 0.1, 0.1)

    for (in_p, gt_p, lab) in tqdm.tqdm(zip(inp_paths, gt_paths, labels)):
        f.write("\t".join(map(str, [in_p, gt_p, lab])) + "\n")
