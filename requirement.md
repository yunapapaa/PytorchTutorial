Environment : CUDA Version: 12.2


1.  Create new environment in anaconda.

`conda create -n 'torch_tuto'`


2. Install required packeges.

```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install conda-forge::matplotlib

conda install pandas
```

PyTorchのversionはCUDAのVersionに応じて適宜変更する必要がある．
`nvidia-smi` でCUDAのVersioを確認し，https://pytorch.org/get-started/previous-versions/ から自分の環境に合ったインストールのコマンドを確認すると良い．


** Note **

After installing, check whther GPU is available with pytorch.

When you run `src/main.py`, 
`CUDA is available: True` will be outputted.

If True, you can use GPU without any problem and the installation was successful !
