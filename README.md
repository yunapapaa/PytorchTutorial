# Pytorch tutorial with CIFAR-10
このコードの軽い説明．(引用はChat GPT)

## 準備
requiment.txtに従って，anaconda環境に必要なパッケージをinstall

## 実行
src/main.pyの冒頭にあるdataset_pathとsave_dirにpathを指定して`python src/main.py`で実行
save_dirは学習結果などの出力グラフを保存するdir
mian.pyを実行すると，
- ex_img.png (どんな画像を学習させたかの例)
- grapgh.png (学習曲線の様子)
  
が出力される

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

main.pyの冒頭にモデルの挙動に変化が見られやすいようなハイパラの設定を明示的に書いてあるため，これらを変更して，モデルの学習の様子やテスト時の識別精度を比較すると良い．

```
n_epoch = 200
lr = 0.1
weight_decay = 0.0
label_smooth = 0.0
lr_scheduling = False
```

過学習を防ぐための工夫をせず (データ拡張もなし)に学習すると，以下のような過学習が観察できる．
<img width="541" alt="image" src="https://github.com/user-attachments/assets/4210a624-072f-47cf-b160-8d5868ed87a5" />


Trainデータに対するLossは下がっているにも関わらず，Validationデータに対するLossは増加している．

このような過学習の問題に対しては，以下の設定を変えてみると効果的である．

- weight decay (とりあえず，1e-5に設定してみるのがおすすめ)
  これは，モデルのパラメータの値が大きくなりすぎないように制約をかけるもの
  
  > weight decay（L2正則化）は、モデルの重みが大きくなりすぎないようにペナルティを課すため、結果的にモデルが非常に大きな重みを使って複雑なパターンを表現する能力を抑制します。これは、過学習を防ぐために有効ですが、一方でモデルの表現力に対して一定の制約をかけることにもなります。適切なweight decayの設定は、表現力と汎化性能のバランスをとるための重要なハイパーパラメータとなります。
  
- label smoothing
  > Label smoothingはこの硬いone-hot表現を少し柔らかくして、正解以外のクラスにもわずかな確率を割り当てることで、モデルが過度に自信を持つのを防ぐ手法です。
  
  one-hot表現はモデルの正解ラベルを候補のクラス数と等しい長さを持つベクトルで表すもの．正解クラスに1.0，その他のクラスが0.0になり，モデルはこのような出力をするように学習する．

- データ拡張
  > データ拡張とは、既存の画像に対して回転や拡大、反転などの変換を加え、学習用データを人工的に増やしてモデルの汎化性能を高める技術です。
  
  src/dataloader.pyの`augmentation_list = ['rcrop', 'hflip', 'cutout']`でさまざまなデータ拡張を指定できる．
  
  CIFAR-10でのおすすめは，`['rcrop', 'hflip', 'cutout']` か，`['rcrop', 'hflip', 'ra', 'cutout']`．

  また，手法を変えるだけでなく，各手法のハイパラを変えてデータ拡張の強度を変えてみてもおもしろい．

  どんなデータ拡張が適用されたかは，指定したsave_dirに出力されるex_img.pngをみると確認できる．


