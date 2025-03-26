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


# ダウンロードして解凍
def download_and_extract_cifar10(save_dir):
    # 保存先ディレクトリが存在しない場合は作成
    os.makedirs(save_dir, exist_ok=True)
    
    # ダウンロードコマンド
    download_command = f"wget -P {save_dir} https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    subprocess.run(download_command, shell=True)
    
    # ダウンロードしたtar.gzファイルのパス
    tar_file = os.path.join(save_dir, 'cifar-10-python.tar.gz')

    # 抽出コマンド（指定したディレクトリに抽出）
    extract_command = f"tar -xf {tar_file} -C {save_dir}"
    subprocess.run(extract_command, shell=True)


# ダウンロードして解凍
def download_and_extract_cifar100(save_dir):
    # 保存先ディレクトリが存在しない場合は作成
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

    # # 保存するdirのpathを定義, 先頭にスラッシュがあると絶対パス扱いになるので除去
    out_dir = os.path.join(output_dir_path, f"cifar10/{phase}/{label}")
    os.makedirs(out_dir, exist_ok=True)

    # 32 x 32にリサイズ
    img = data.reshape(3, 32, 32)
    # (C, H, W) -> (H, W, C) に並び替え
    img = np.transpose(img, (1, 2, 0))

    # 画像を保存
    img = Image.fromarray(img)
    img.save(os.path.join(out_dir, name.decode('utf-8')))


def save_image_parallel_cifar100(pool_list):
    label, data, name, output_dir_path, phase = pool_list

    # 保存するdirのpathを定義
    out_dir = os.path.join(output_dir_path, f"cifar100/{phase}/{label}")
    os.makedirs(out_dir, exist_ok=True)

    # 32 x 32にリサイズ
    img = data.reshape(3, 32, 32)
    # (C, H, W) -> (H, W, C) に並び替え
    img = np.transpose(img, (1, 2, 0))

    # 画像を保存
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

    # 並列処理のための入力データリストを初期化
    pool_list = []
    for path2file in file_path_list:
        # pickleで保存された辞書型データを読み込む（バイナリキー付き）
        with open(path2file, "rb") as f:
            dict_data = pickle.load(f, encoding="bytes")

            # ラベル（fine_labels), 画像データ（data), ファイル名（filenames）を取り出し，並列処理用リストに追加
            for label, data, name in zip(dict_data[b"labels"], dict_data[b"data"], dict_data[b"filenames"]):
                pool_list.append([label, data, name, output_dir_path, phase])

    # 8プロセスでをsave_image_parallel_cifar10を並列に実行
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

    # 並列処理のための入力データリストを初期化
    pool_list = []

    # CIFAR-100のpickleファイルを読み込む
    for path2file in file_path_list:
        # pickleで保存された辞書型データを読み込む（バイナリキー付き）
        with open(path2file, "rb") as f:
            dict_data = pickle.load(f, encoding="bytes")
            
            # for key in dict_data:
            #     print(key)

            # ラベル（fine_labels), 画像データ（data), ファイル名（filenames）を取り出し，並列処理用リストに追加
            for label, data, name in zip(dict_data[b"fine_labels"], dict_data[b"data"], dict_data[b"filenames"]):
                pool_list.append([label, data, name, output_dir_path, phase])

    # 8プロセスでをsave_image_parallel_cifar100を並列に実行
    p = Pool(8)
    p.map(save_image_parallel_cifar100, pool_list)
    p.close()
    p.join()


# valに選ばれたデータをtrain dirからval dirに移す
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

    # 各クラスごとにtrainとvalのサンプル数を計算
    class_train_size = int(np.floor(train_size / n_class))
    class_val_size = int(np.floor(val_size / n_class))

    # val dataを保存するdirを作成
    out_dir = os.path.join(output_dir_path, name, "val")
    os.makedirs(out_dir, exist_ok=True)
    
    print(len(["train size for a class" for _ in range(class_train_size)]))
    print(len(["val size for a class" for _ in range(class_val_size)]))

    # 各サンプルごとに，"train" または "val" のラベルを付けたリストを作成．ランダムにシャッフル
    phase_list = ["train" for _ in range(class_train_size)] + ["val" for _ in range(class_val_size)]
    random.shuffle(phase_list)

    # 並列処理で使用する入力のリストを初期化
    pool_list = []
    for class_id in range(n_class):
        # 各クラスごとのval用dirのパス，dirを作成
        class_val_dir = os.path.join(out_dir, str(class_id))
        os.makedirs(class_val_dir, exist_ok=True)

        # 各クラスのtrainのdirから，pngファイルのパスを全て取得
        img_list = sorted(glob.glob(os.path.join(output_dir_path, name, "train", str(class_id), "*.png")))
        count = 0

        # 画像のパスとphase_listのラベルを対応させ，val用に処理する画像を選定
        for path2img, phase in zip(img_list, phase_list):
            # valに選ばれたら，対象の画像パス，クラスID，出力dirの情報を追加する
            if phase == "val":
                count += 1
                pool_list.append([path2img, class_id, out_dir])
                
        print(f"for {class_id}, n_val: {count}")

    # 8プロセスのプールを作成，並列でsplit_train_val_parallel関数を実行
    p = Pool(8)
    p.map(split_train_val_parallel, pool_list)
    p.close()
    p.join()


# データセット分割の割合とseed値を記録
def save_log(output_dir_path, r_seed, name, train_size, val_size):
    out_path = os.path.join(output_dir_path, name, "log.txt")
    with open(out_path, "w") as f:
        f.write(f"Create: {datetime.datetime.now()}\n")
        f.write(f"Train:val={train_size}:{val_size}, randomseed={r_seed}")


# CIFAR10用
def create_CIFAR10_dataset(output_dir_path, r_seed, val_size):
    train_size = 50000 - val_size
    
    np.random.seed(r_seed)
    random.seed(r_seed)

    # データをダウンロードして解凍
    download_and_extract_cifar10(output_dir_path)

    # trainを保存
    save_cifar10_images(output_dir_path, phase="train")

    # testを保存
    save_cifar10_images(output_dir_path, phase="test")

    # trainをtrain, valに分割
    make_val(output_dir_path, "cifar10", train_size, val_size)

    # データセット作成時の条件を保存
    save_log(output_dir_path, r_seed, "cifar10", train_size, val_size)


# CIFAR100用
def create_CIFAR100_dataset(output_dir_path, r_seed, val_size):
    train_size = 50000 - val_size
    
    np.random.seed(r_seed)
    random.seed(r_seed)

    # データをダウンロードして解凍
    download_and_extract_cifar100(output_dir_path)

    # trainを保存
    save_cifar100_images(output_dir_path, phase="train")

    # testを保存
    save_cifar100_images(output_dir_path, phase="test")

    # trainをtrain, valに分割
    make_val(output_dir_path, "cifar100", train_size, val_size)

    # データセット作成時の条件を保存
    save_log(output_dir_path, r_seed, "cifar100", train_size, val_size)


print(f"Create {DATASET} at {DATASET_PATH}")
if DATASET == "cifar10":
    create_CIFAR10_dataset(DATASET_PATH, 1, 10000)
elif DATASET == "cifar100":
    create_CIFAR100_dataset(DATASET_PATH, 1, 10000)
