import torch.nn as nn
from torchvision.models.resnet import BasicBlock

class ControlResNet(nn.Module):
    """
    ResNet for CIFAR-10 that mirrors AKOrNResNet’s tail:
      conv stem → [BasicBlock×2] × 2 → GAP → FC
    Width (=base_dim) を調整して総パラメータ数を合わせる。
    """
    def __init__(
            self,
            base_dim=128,
            num_classes=10,
            blocks=3
            ):
        super().__init__()

        # stem: 3×3 conv（Stride1）
        # self.conv1 = nn.Conv2d(3, base_dim, 3, 1, 1, bias=False)
        # self.bn1   = nn.BatchNorm2d(base_dim)
        # self.relu  = nn.ReLU(inplace=True)

        def make_layer(in_ch, out_ch, stride, blocks: int=2):
            # ① 先頭ブロック用の downsample を準備
            downsample = None
            if stride != 1 or in_ch != out_ch:
                downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

            # ② ブロックを積む
            layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_ch, out_ch))  # 2 枚目以降 stride=1
            return nn.Sequential(*layers)

        self.layer0 = make_layer(       3,     base_dim, 1, blocks)
        self.layer1 = make_layer(base_dim,   2*base_dim, 2, 2)  # 32→16
        self.layer2 = make_layer(2*base_dim, 4*base_dim, 2, 2)  # 16→8
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(4*base_dim, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        #x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
