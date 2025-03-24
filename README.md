# Pytorch tutorial with CIFAR-10, CIFAR-100
このコードの軽い説明．(引用はChat GPT)

## 準備
- ライブラリ
  
  requiment.mdを参照して，anaconda環境に必要なパッケージをinstall

- データセット
  
  `src/create_dataset.py`の冒頭部分で，DATASETにデータセット名を指定．(CIFAR-10なら，'cifar10', CIFAR-100なら，'cifar100')
  `DATASET_PATH`にデータセットを保存するディレクトリのパスを指定．

  `python src/create_daset.py` で実行．
  
  以下のようなディレクトリの構造のデータセットが作成される．
    
    ```
    cifar10/
    └── train/
        ├── 0/
          ├── aeroplane_s_000004.png
          ├── aeroplane_s_000021.png
          └── ...
    
        ├── 1/
        ├── 2/
        └── ...
    
    └── val/
        ├── 0/   
        ├── 1/
        └── ...
    
    └── test/
        ├── 0/   
        ├── 1/
        └── ...
    ```

    デフォルトの状態では，学習データを，Train data : Validation data = 40000 : 10000 に分割するが，`create_CIFAR10_dataset(DATASET_PATH, r_seed=1, val_size=10000)`でValidation dataの数を指定できる．


## 実行
src/main.pyの冒頭で，dataset_path, save_dirのパスを指定．

dataset_pathは，`dataset_path = '/path_to_dataset/cifar10'`のようにデータセットのディレクトリのパスを指定．
save_dirは，学習結果などの出力グラフを保存するためのディレクトリのパスを指定．

`python src/main.py` で実行.

mian.pyを実行すると，
- ex_img.png (どんな画像を学習させたかの例)
- grapgh.png (学習曲線の様子)
  
がsave_dirに出力される．

## モデル
- My CNN
  
  畳み込み層1, 畳み込み層2, FCからなる，3層の簡単な畳み込みニューラルネットワークモデル

- ResNet
  
  有名なCNNモデル
  > ResNetは、各層の出力を次の層に直接伝える「残差接続」により、非常に深いネットワークでもスムーズに学習できる点がすごいです。
  
  実装しているResNetは，CIFAR-10のような小さい画像 (解像度：32x32)に対応したResNetなので，通常のResNetとは少し違う．

  `model = ResNetBasicBlock(depth=20, n_class=10)`で定義するときに，depth = 20, 56のように設定すると，ResNet20, ResNet56が使える．

  まず，model/my_cnn.pyを見て，CNNってPytorchでこんなふうに定義するんだというのを確認できたら，ResNet20を使ってみるのがおすすめ．

## ハイパーパラメータ

main.pyの冒頭にモデルの挙動に変化が見られやすいようなハイパラの設定を明示的に書いてある．これらを変更して，モデルの学習の様子やテスト時の識別精度を比較すると良い．

```
n_epoch = 200
lr = 0.1
weight_decay = 0.0
label_smooth = 0.0
lr_scheduling = False
```

## Overfitting(過学習)
ハイパラ設定を変更する際には，過学習を抑制することで，モデルの識別精度が向上する様子を確認できると勉強になる．

過学習を防ぐための工夫をせず(上のハイパラ設定のまま + データ拡張なし)で学習すると，以下のような過学習が観察できる．(MyCNN, CIFAR-100)
![w_overfit](https://github.com/user-attachments/assets/0ecc47a0-5fc2-4c2b-a519-391f943cce1f)


30エポックあたりでValidation Acuracyは最大となり，Trainデータに対するLossが下がっているにも関わらず，Validationデータに対するLossが増加している．


このような過学習の問題に対しては，以下の設定を変えて，学習の工夫を行ってみると効果的である．

- weight decay 
  これは，モデルのパラメータの値が大きくなりすぎないように制約をかけるもの
  
  > weight decay（L2正則化）は、モデルの重みが大きくなりすぎないようにペナルティを課すため、結果的にモデルが非常に大きな重みを使って複雑なパターンを表現する能力を抑制します。これは、過学習を防ぐために有効ですが、一方でモデルの表現力に対して一定の制約をかけることにもなります。適切なweight decayの設定は、表現力と汎化性能のバランスをとるための重要なハイパーパラメータとなります。

  とりあえず，1e-5に設定してみるのがおすすめ．
  
- label smoothing
  > Label smoothingはこの硬いone-hot表現を少し柔らかくして、正解以外のクラスにもわずかな確率を割り当てることで、モデルが過度に自信を持つのを防ぐ手法です。
  
  one-hot表現はモデルの正解ラベルを候補のクラス数と等しい長さを持つベクトルで表すもの．正解クラスに1.0，その他のクラスが0.0になり，モデルはこのような出力をするように学習する．

- データ拡張
  > データ拡張とは、既存の画像に対して回転や拡大、反転などの変換を加え、学習用データを人工的に増やしてモデルの汎化性能を高める技術です。
  
  src/dataloader.pyの`augmentation_list = ['rcrop', 'hflip', 'cutout']`でさまざまなデータ拡張を指定できる．
  
  CIFAR-10でのおすすめは，`['rcrop', 'hflip', 'cutout']` か，`['rcrop', 'hflip', 'ra', 'cutout']`．

  また，手法を変えるだけでなく，各手法のハイパラを変えてデータ拡張の強度を変えてみてもおもしろい．

  どんなデータ拡張が適用されたかは，指定したsave_dirに出力されるex_img.pngをみると確認できる．



(参考) Weight Decay, Label Smooting, データ拡張を加えて学習させると以下のようになる．(MyCNN, CIFAR-100)

  ![wo_overfit](https://github.com/user-attachments/assets/0d985c44-21f9-43b0-99d1-fcda0aa0bbc8)




