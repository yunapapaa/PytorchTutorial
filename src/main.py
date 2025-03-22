import time
import os
import numpy as np
import pandas as pd
import sys

import torch
import torch.nn as nn


from dataloader import get_dataloader
from train import train_model, accuracy
from utils.common import setup_device, fixed_r_seed, get_time
from model.cifar_resnet import ResNetBasicBlock
from model.my



def main():
    print('start main')

    SEED=1
    N_CLASS=10
    N_EPOCH = 10
    dataset_path = '/homes/ypark/code/working_dataset/cifar10'
    
    device = setup_device()
    fixed_r_seed(SEED)

    # ImageNetのvalidationデータ19クラス．各クラス50枚．224にリサイズ．
    val_dataloader = get_dataloader(dataset_path=dataset_path, img_size=224, batch_size=128)
    
    # ImageNetで学習済みモデル
    model = ResNetBasicBlock(weights = "ViT_B_16_Weights.IMAGENET1K_V1")
    model.heads[0] = nn.Linear(model.heads[0].in_features, N_CLASS)
    model.to(device)

    criterion = loss = nn.CrossEntropyLoss(reduction='mean')

    # 分類層だけチューニングする場合
    # train_model(device, N_EPOCH, criterion, model, val_dataloader)
    
    # モデルに入力
    model.eval()
    test_acc, test_loss, n_test = 0, 0, 0
    with torch.no_grad():
        for i, sample_batched in enumerate(val_dataloader):
            data, target = sample_batched["image"].to(device), sample_batched["label"].to(device)
            output = model(data)
            loss = criterion(output, target)
            test_acc += accuracy(output, target)
            test_loss += loss.item() * target.size(0)
            n_test += target.size(0)
    
    test_loss = test_loss / n_test
    test_acc = float(test_acc) / n_test

    print(f'Test Loss : {test_loss:.6f}  Test Accuracy : {test_acc * 100:.2f}%')

    


if __name__ == "__main__":
    main()





