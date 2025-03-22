import torch


def accuracy(output, target):
    pred = output.data.max(1, keepdim=True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()
    return acc


# Training
def train(model, device, train_loader, optimizer, criterion):
    # trainモードに設定
    model.train()

    # 1バッチずつ処理
    train_acc, train_loss, n_train = 0, 0, 0
    for i_batch, sample_batched in enumerate(train_loader):
        data, target = sample_batched["image"].to(device), sample_batched["label"].to(device)

        # モデルの出力, 損失を計算
        output = model(data)
        loss = criterion(output, target)

        # パラメータを更新
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        # バッチ内における学習進捗を記録
        train_acc += accuracy(output, target)
        train_loss += loss.item() * target.size(0)
        n_train += target.size(0)

    # 全バッチの平均loss, accを返す
    return train_loss / n_train, float(train_acc) / n_train


# Validation
def val(model, device, val_loader, criterion):
    # evaluationモードに設定
    model.eval()

    # 1バッチずつ処理
    val_acc, val_loss, n_val = 0, 0, 0
    # 勾配計算はしない
    with torch.no_grad():
        for i, sample_batched in enumerate(val_loader):
            data, target = sample_batched["image"].to(device), sample_batched["label"].to(device)

            # モデルの出力, 損失を計算
            output = model(data)
            loss = criterion(output, target)

            # バッチ内における推論結果を記録
            val_acc += accuracy(output, target)
            val_loss += loss.item() * target.size(0)
            n_val += target.size(0)

    # 全バッチの平均loss, accを返す
    return val_loss / n_val, float(val_acc) / n_val



# Test
def test(model, device, test_loader, criterion):
    model.eval()
    test_acc, test_loss, n_test = 0, 0, 0

    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            data, target = sample_batched["image"].to(device), sample_batched["label"].to(device)

            output = model(data)
            loss = criterion(output, target)

            test_acc += accuracy(output, target)
            test_loss += loss.item() * target.size(0)
            n_test += target.size(0)

    return test_loss / n_test, float(test_acc) / n_test


