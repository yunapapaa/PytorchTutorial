import torch


def accuracy(output, target):
    pred = output.data.max(1, keepdim=True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()
    return acc


def train_model(device, n_epoch, criterion, model, train_loader):
     # 分類層だけチューニングする場合
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads[0].parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-3, weight_decay=0.0)

    # 学習
    for epoch in range(1, n_epoch + 1):
        train_acc, train_loss, n_train = 0, 0, 0
        # train
        model.train()
        for i_batch, sample_batched in enumerate(train_loader):
            data, target = sample_batched["image"].to(device), sample_batched["label"].to(device)
            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            train_acc += accuracy(output, target)
            train_loss += loss.item() * target.size(0)
            n_train += target.size(0)

        train_loss = train_loss / n_train
        train_acc = float(train_acc) / n_train

        print(f'Epoch: [{epoch:03}/{n_epoch:03}]  Train Loss: {train_loss:.6f}  Train Accuracy: {train_acc * 100:.2f}%')
    