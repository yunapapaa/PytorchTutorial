import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

# arial_path = '/homes/ypark/fonts/Arial.ttf'  # 自分のユーザーディレクトリのパスに合わせて修正

# # フォントを登録
# fm.fontManager.addfont(arial_path)

# # フォントファミリを Arial に設定
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams["pdf.fonttype"] = 42  # PDF出力時に TrueType フォント（Type 42）を使用
# plt.rcParams["ps.fonttype"] = 42


# train, val, leave_oneのlossをplot
def plot_loss(save_path, data):
    # dfの最後のtest loss, test accはplotしない
    ex_col = 2

    epochs = np.arange(1, len(data) + 1 - ex_col)
    fig, ax = plt.subplots(2, 1, figsize=(15, 15))
    
    # lossのplot
    ax[0].plot(epochs, data["train_loss"][:-ex_col], label='Train', alpha=0.8, linewidth=3, marker='o')
    ax[0].plot(epochs, data["val_loss"][:-ex_col], label='Validation', alpha=0.8, linewidth=3, marker='o')

    ax[0].set_title('Loss', fontsize=30)
    ax[0].set_xlabel('Epochs', fontsize=25)
    ax[0].set_ylabel('Loss', fontsize=25)
    ax[0].legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.2, fontsize=20, ncol=1)
    ax[0].tick_params(labelsize=25)
    ax[0].grid()

    max_index = data["val_acc"][:-ex_col].idxmax()
    max_acc = round(data["val_acc"][:-ex_col].loc[max_index] * 100, 2)

    # accのplot
    ax[1].plot(epochs, data["train_acc"][:-ex_col]*100, label='Training', alpha=0.8, linewidth=3, marker='o')
    ax[1].plot(epochs, data["val_acc"][:-ex_col]*100, label='Validation', alpha=0.8, linewidth=3, marker='o')
    
    ax[1].axvline(x=max_index + 1, color='b', linewidth=2, alpha=0.5)

    ax[1].set_title(f'Accuracy (Best Val : {max_acc}%)', fontsize=30)
    ax[1].set_xlabel('Epochs', fontsize=25)
    ax[1].set_ylabel('Accuracy', fontsize=25)
    # ax[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[1].set_ylim(50, 100)
    ax[1].legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0.2, fontsize=20, ncol=1)
    ax[1].tick_params(labelsize=25)
    ax[1].grid()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
