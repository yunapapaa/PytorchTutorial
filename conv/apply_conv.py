from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

### (参考)これは畳み込み演算の様子を見るためのコード ###

# 2d畳み込みフィルタを適用
def conv2d(image, kernel):
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    out = np.zeros((img_h - ker_h + 1, img_w - ker_w + 1), dtype=np.float32)

    # 畳み込み演算のためにカーネルを180度反転
    flipped_kernel = np.flipud(np.fliplr(kernel)) 

    # カーネルを1ピクセルずつずらすながら適用
    for i in range(img_h - ker_h + 1):
        for j in range(img_w - ker_w + 1):
            region = image[i:i+ker_h, j:j+ker_w]
            out[i, j] = np.sum(region * flipped_kernel)
    
    return out

# カーネルを可視化するために，0-255にスケール
def normalize_kernel_for_vis(kernel):
    k_vis = (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255
    return k_vis.astype(np.uint8)


# 画像を読み込む 
img = Image.open('/homes/ypark/code/torch_tuto/conv/lena_std.png')
# グレースケール化
img = img.convert('L')
img = np.array(img, dtype=np.float32)

print(f'Original img shape : {img.shape}')


## 畳み込みカーネル ##
# 例として，代表的なフィルタ3種類
# (CNNはこの値を重みとして学習)

# 横方向のエッジ（Sobel Y）
kernel_1 = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# 縦方向のエッジ（Sobel X）
kernel_2 = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float32)

# 全方向エッジ（Laplacian）
kernel_3 = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# 畳み込みカーネルを適用
filtered_img_1 = conv2d(img, kernel_1)
filtered_img_2 = conv2d(img, kernel_2)
filtered_img_3 = conv2d(img, kernel_3)

# 値を[0, 255]にクリップしてuint8に変換
filtered_img_1 = np.clip(filtered_img_1, 0, 255).astype(np.uint8)
filtered_img_2 = np.clip(filtered_img_2, 0, 255).astype(np.uint8)
filtered_img_3 = np.clip(filtered_img_3, 0, 255).astype(np.uint8)


# 3*3のカーネルをストライド1で適用するとタテヨコ 2 pixel 分小さくなる
print(f'Converted img shape : {filtered_img_1.shape}')

filtered_imgs = [filtered_img_1, filtered_img_2, filtered_img_3]
# カーネルを可視化用に0-255スケールに
kernel_vis = [normalize_kernel_for_vis(k) for k in [kernel_1, kernel_2, kernel_3]]

# カーネルと適用後の画像を可視化
fig, axs = plt.subplots(3, 3, figsize=(10, 12))

for i in range(3):
    axs[i, 0].imshow(img, cmap='gray')
    axs[i, 0].set_title(f'Original')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(kernel_vis[i], cmap='gray')
    axs[i, 1].set_title(f'Kernel {i+1}')
    axs[i, 1].axis('off')

    axs[i, 2].imshow(filtered_imgs[i], cmap='gray')
    axs[i, 2].set_title(f'Filtered Image {i+1}')
    axs[i, 2].axis('off')

plt.tight_layout()
plt.savefig("/homes/ypark/code/torch_tuto/conv/result.png")