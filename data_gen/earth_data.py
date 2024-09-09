# https://github.com/DeokyunKim/Progressive-Face-Super-Resolution/blob/master/dataloader.py
import cv2
import os
import sys
import traceback
import numpy as np

from tqdm import tqdm

from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder


def load_file(path):

    # always return [H, W, C]
    if path.endswith('.jpg') or path.endswith('.png'):
        img = cv2.imread(path)
        return np.flip(img, 2)# RGB Format!
    elif path.endswith('.npy'):
        return np.load(path).transpose(1, 2, 0)
    else:
        print("ERROR: File Extension '{}' could not be parsed!".format(path.split('.')[-1]))
        sys.exit(-1)


def build_bin_dataset(imgs, prefix):
    raw_data_dir_inp    = hparams['raw_data_dir_inp']
    raw_data_dir_gt     = hparams['raw_data_dir_gt']

    in_size             = hparams['in_size']
    scale_factor        = hparams['sr_scale']
    gt_size             = in_size * scale_factor

    binary_data_dir     = hparams['binary_data_dir']
    os.makedirs(binary_data_dir, exist_ok=True)
    builder = IndexedDatasetBuilder(f'{binary_data_dir}/{prefix}')
    for (img_lr, img_hr) in tqdm(imgs):
        try:
            base_name       = img_lr.split('.')[:-1]

            full_path_lr    = f'{raw_data_dir_inp}/{img_lr}'
            full_path_hr    = f'{raw_data_dir_gt}/{img_hr}'

            data_lr         = load_file(full_path_lr)
            data_hr         = load_file(full_path_hr)

            data_lr         = cv2.resize(data_lr, (in_size, in_size)).astype(np.float32)
            data_hr         = cv2.resize(data_hr, (gt_size, gt_size)).astype(np.float32)

            builder.add_item({'item_name': base_name, 'img_lr': data_lr, 'img_hr': data_hr})
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            print("| binarize img error: ", img_lr, img_hr)
    builder.finalize()


if __name__ == '__main__':
    set_hparams()

    eval_partition_path = hparams['set_table']

    train_img_list  = []
    val_img_list    = []
    test_img_list   = []
    with open(eval_partition_path, mode='r') as f:
        while True:
            line = f.readline().split()
            if not line: break
            if line[2] == '0':
                train_img_list.append(line[:2])
            elif line[2] == '1':
                val_img_list.append(line[:2])
            else:
                test_img_list.append(line[:2])

    build_bin_dataset(train_img_list, 'train')
    build_bin_dataset(val_img_list, 'valid')
    build_bin_dataset(test_img_list, 'test')
