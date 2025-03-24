import datetime
import glob
import os
import pickle
import random
import subprocess
from multiprocessing import Pool

import numpy as np
from PIL import Image

DATASET = 'cifar10'
DATASET_PATH = '/homes/ypark/code/dataset/test'

def download_and_extract_cifar10(save_dir):
    # 保存先ディレクトリが存在しない場合は作成する
    os.makedirs(save_dir, exist_ok=True)
    
    # ダウンロードコマンド（必要に応じてコメント解除）
    download_command = f"wget -P {save_dir} https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    subprocess.run(download_command, shell=True)
    
    # ダウンロードしたtar.gzファイルのパス
    tar_file = os.path.join(save_dir, 'cifar-10-python.tar.gz')

    # 抽出コマンド（指定したディレクトリに抽出）
    extract_command = f"tar -xf {tar_file} -C {save_dir}"
    subprocess.run(extract_command, shell=True)


def download_and_extract_cifar100(save_dir):
    # 保存先ディレクトリが存在しない場合は作成する
    os.makedirs(save_dir, exist_ok=True)
    
    # ダウンロードコマンド
    download_command = f"wget -P {save_dir} https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    subprocess.run(download_command, shell=True)
    
    # ダウンロードしたtar.gzファイルのパス
    tar_file = os.path.join(save_dir, 'cifar-100-python.tar.gz')
    
    # 抽出コマンド（指定したディレクトリに抽出）
    extract_command = f"tar -xf {tar_file} -C {save_dir}"
    subprocess.run(extract_command, shell=True)


def save_image_parallel_cifar10(pool_list):
    label, data, name, output_dir_path, phase = pool_list

    # 先頭にスラッシュがあると絶対パス扱いになるので除去
    out_dir = os.path.join(output_dir_path, f"cifar10/{phase}/{label}")
    os.makedirs(out_dir, exist_ok=True)

    img = data.reshape(3, 32, 32)
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    img.save(os.path.join(out_dir, name.decode('utf-8')))


def save_image_parallel_cifar100(pool_list):
    label, data, name, output_dir_path, phase = pool_list

    out_dir = os.path.join(output_dir_path, f"cifar100/{phase}/{label}")
    os.makedirs(out_dir, exist_ok=True)

    img = data.reshape(3, 32, 32)
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    img.save(os.path.join(out_dir, name.decode('utf-8')))


def save_cifar10_images(output_dir_path, phase):
    # phase: "train" または "test"
    if phase == "train":
        file_path_list = [f"{output_dir_path}/cifar-10-batches-py/data_batch_{idx}" for idx in range(1, 6)]
    elif phase == "test":
        file_path_list = [f"{output_dir_path}/cifar-10-batches-py/test_batch"]
    else:
        raise ValueError("no such phase!")

    pool_list = []
    for path2file in file_path_list:
        with open(path2file, "rb") as f:
            dict_data = pickle.load(f, encoding="bytes")
            for label, data, name in zip(dict_data[b"labels"], dict_data[b"data"], dict_data[b"filenames"]):
                pool_list.append([label, data, name, output_dir_path, phase])
    p = Pool(8)
    p.map(save_image_parallel_cifar10, pool_list)
    p.close()
    p.join()


def save_cifar100_images(output_dir_path, phase):
    # phase: "train" または "test"
    if phase == "train":
        file_path_list = [os.path.join(output_dir_path, "cifar-100-python", "train")]
    elif phase == "test":
        file_path_list = [os.path.join(output_dir_path, "cifar-100-python", "test")]
    else:
        raise ValueError("no such phase!")

    pool_list = []
    for path2file in file_path_list:
        with open(path2file, "rb") as f:
            dict_data = pickle.load(f, encoding="bytes")
            # 必要に応じてデバッグ用のキー表示
            # for key in dict_data:
            #     print(key)
            for label, data, name in zip(dict_data[b"fine_labels"], dict_data[b"data"], dict_data[b"filenames"]):
                pool_list.append([label, data, name, output_dir_path, phase])
    p = Pool(8)
    p.map(save_image_parallel_cifar100, pool_list)
    p.close()
    p.join()


def split_train_val_parallel(pool_list):
    path2img, class_id, out_dir = pool_list
    command = f"mv {path2img} {out_dir}/{class_id}/"
    subprocess.run(command, shell=True)


def make_val(output_dir_path, name, train_size, val_size):
    print(f"split Train:Val = {train_size}:{val_size}")
    if name == "cifar10":
        n_class = 10
    elif name == "cifar100":
        n_class = 100
    else:
        raise ValueError("Invalid dataset name.")

    class_train_size = int(np.floor(train_size / n_class))
    class_val_size = int(np.floor(val_size / n_class))
    out_dir = os.path.join(output_dir_path, name, "val")
    os.makedirs(out_dir, exist_ok=True)
    print(len(["train size for a class" for _ in range(class_train_size)]))
    print(len(["val size for a class" for _ in range(class_val_size)]))
    phase_list = ["train" for _ in range(class_train_size)] + ["val" for _ in range(class_val_size)]
    random.shuffle(phase_list)

    pool_list = []
    for class_id in range(n_class):
        class_val_dir = os.path.join(out_dir, str(class_id))
        os.makedirs(class_val_dir, exist_ok=True)
        img_list = sorted(glob.glob(os.path.join(output_dir_path, name, "train", str(class_id), "*.png")))
        count = 0
        for path2img, phase in zip(img_list, phase_list):
            if phase == "val":
                count += 1
                pool_list.append([path2img, class_id, out_dir])
        print(f"for {class_id}, n_val: {count}")
    p = Pool(8)
    p.map(split_train_val_parallel, pool_list)
    p.close()
    p.join()


def save_log(output_dir_path, r_seed, name, train_size, val_size):
    out_path = os.path.join(output_dir_path, name, "log.txt")
    with open(out_path, "w") as f:
        f.write(f"Create: {datetime.datetime.now()}\n")
        f.write(f"Train:val={train_size}:{val_size}, randomseed={r_seed}")


def create_CIFAR10_dataset(output_dir_path, r_seed, val_size):
    train_size = 50000 - val_size
    np.random.seed(r_seed)
    random.seed(r_seed)
    download_and_extract_cifar10(output_dir_path)

    save_cifar10_images(output_dir_path, phase="train")
    save_cifar10_images(output_dir_path, phase="test")
    make_val(output_dir_path, "cifar10", train_size, val_size)

    save_log(output_dir_path, r_seed, "cifar10", train_size, val_size)


def create_CIFAR100_dataset(output_dir_path, r_seed, val_size):
    train_size = 50000 - val_size
    np.random.seed(r_seed)
    random.seed(r_seed)
    download_and_extract_cifar100(output_dir_path)

    save_cifar100_images(output_dir_path, phase="train")
    save_cifar100_images(output_dir_path, phase="test")
    make_val(output_dir_path, "cifar100", train_size, val_size)

    save_log(output_dir_path, r_seed, "cifar100", train_size, val_size)


print(f"Create {DATASET} at {DATASET_PATH}")
if DATASET == "cifar10":
    create_CIFAR10_dataset(DATASET_PATH, 1, 10000)
elif DATASET == "cifar100":
    create_CIFAR100_dataset(DATASET_PATH, 1, 10000)
