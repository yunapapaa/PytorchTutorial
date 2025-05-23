import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.convs(x)
        if self.downsample is not None:
            # convによって特徴マップの数，大きさが変化した分を残差も修正
            # [bs, ch, h, w]  -> [bs, 2 * ch , h / 2,  w / 2])
            residual = self.downsample(x)

        # スキップ接続
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.convs(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        # スキップ接続
        out += residual
        out = self.relu(out)
        return out
    
# 20, 
class ResNetBasicBlock(nn.Module):
    def __init__(self, depth=20, n_class=10):
        super().__init__()
        # 指定した深さ（畳み込みの層数）でネットワークを構築できるかを確認
        assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2 (e.g. 20, 32, 44).'
        n_blocks = (depth - 2) // 6  # 1ブロックあたりのBasic Blockの数を決定

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # layer1はstride=1(特徴マップのサイズに変化なし)，ch数も16 -> 16
        self.layer1 = self._make_layer(16, n_blocks)
        # stride=2なので特徴マップは半分のサイズへ
        self.layer2 = self._make_layer(32, n_blocks, stride=2)
        self.layer3 = self._make_layer(64, n_blocks, stride=2)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * BasicBlock.expansion, n_class)

    def _make_layer(self, planes, n_blocks, stride=1):
        downsample = None
        # スキップ接続用の1x1畳みこみ
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        # ブロック内での最初のlayerのstrideは可変
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        # 以降のlayerはstride=1
        for _ in range(0, n_blocks - 1):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 入力：[128, 3, 32, 32]

        # conv1 : [128, 3, 32, 32] -> [128, 16, 32, 32]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # layer1 : [128, 16, 32, 32] -> [128, 16, 32, 32]
        x = self.layer1(x)

        # layer2 : [128, 16, 32, 32] -> [128, 32, 16, 16]
        x = self.layer2(x)

        # layer3 : [128, 32, 16, 16] -> [128, 64, 8, 8]
        x = self.layer3(x)

        # gap : [128, 64, 8, 8] -> [128, 64, 1, 1]
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
# 56
class ResNetBottleneck(nn.Module):
    def __init__(self, depth, n_class=10):
        super().__init__()
        # 指定した深さ（畳み込みの層数）でネットワークを構築できるかを確認
        assert (depth - 2) % 9 == 0, 'When use Bottleneck, depth should be 9n+2 (e.g. 47, 56, 110, 1199).'
        n_blocks = (depth - 2) // 9  # 1ブロックあたりのBasic Blockの数を決定

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, n_blocks)
        self.layer2 = self._make_layer(32, n_blocks, stride=2)
        self.layer3 = self._make_layer(64, n_blocks, stride=2)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * Bottleneck.expansion, n_class)

    def _make_layer(self, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(0, n_blocks - 1):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
