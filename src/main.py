import time
import os
import numpy as np
import pandas as pd
import sys

import torch
import torch.nn as nn

from dataloader import get_dataloader
from train_val import train, val, test
from model.my_cnn import MyCNN
from model.cifar_resnet import ResNetBasicBlock
from utils.common import setup_device, fixed_r_seed, get_time, show_img
from utils.plot import plot_loss

print('giselle')


def main():

    seed=1
    n_epoch = 200
    lr = 0.1
    weight_decay = 0.0
    label_smooth = 0.0
    lr_scheduling = False
    dataset_path = '/homes/ypark/code/working_dataset/cifar10'
    save_dir = '/homes/ypark/code/torch_tuto/fig'
    
    # gpu使える場合はcudaを登録
    device = setup_device()
    fixed_r_seed(seed)

    # データセットの読み込み
    train_loader, val_loader, test_loader = get_dataloader(dataset_path=dataset_path, img_size=32, batch_size=128)
    # データ拡張の適用結果を確認
    show_img(save_path=os.path.join(save_dir, 'ex_img.png'), dataloader=train_loader)
    
    ## モデル
    model = MyCNN(n_class=10)
    # model = ResNetBasicBlock(depth=20, n_class=10)
    model.to(device)

    # モデル構造を出力して確認する場合
    # with open(os.path.join(save_dir, 'structure_resnet20.txt'), 'w') as f:
    #     print(model, file=f)

    # 最適化アルゴリズムの定義
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    # 学習率のスケジューラーを設定 (Cosineで init lr * 1/100 まで減衰)
    if lr_scheduling:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=n_epoch,
                eta_min=lr*0.01,
        )

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=label_smooth)

    start = time.time()
    all_training_result = []
    for epoch in range(1, n_epoch + 1):
        interval = time.time() - start
        interval = get_time(interval)
        print(f"Lr: {optimizer.param_groups[0]['lr']} , Time: {interval['time']}")
    
        # train
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)

        # validation
        val_loss, val_acc = val(model, device, val_loader, criterion)

        # 学習進捗を記録
        all_training_result.append([train_loss, train_acc, val_loss, val_acc])
       
        print(
            f"Epoch: [{epoch}/{n_epoch}] \t"
            + f"Train Loss: {train_loss:.6f} \t"
            + f"Train Acc: {train_acc*100:.2f}% \t"
            + f"Val Loss: {val_loss:.6f} \t"
            + f"Val Acc: {val_acc*100:.2f}% \t"
        )
        sys.stdout.flush()

        # 学習率の更新
        if lr_scheduling:
            scheduler.step()
    

    all_training_result = pd.DataFrame(
        np.array(all_training_result),
        columns=["train_loss", "train_acc", "val_loss", "val_acc"],
    )
    interval = time.time() - start
    interval = get_time (interval)

    # テスト
    test_loss, test_acc = test(model, device, test_loader, criterion)
    print(
        f"Time: {interval['time']}  Test loss: {test_loss:.6f}  Test Acc: {test_acc*100:.2f}")

    all_training_result.loc["test_acc"] = test_acc
    all_training_result.loc["test_loss"] = test_loss
    # all_training_result.to_csv(save_file_path, index=False)

    # 学習曲線をplotして確認
    plot_loss(os.path.join(save_dir, "graph.png"), all_training_result)


    
if __name__ == "__main__":
    main()





