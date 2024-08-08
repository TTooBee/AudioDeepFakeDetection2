import torch
import torch.nn as nn

class Residual_block2D(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            padding=1,
            kernel_size=3,
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=0,
                kernel_size=1,
                stride=1,
            )
        else:
            self.downsample = False

        #self.mp = nn.MaxPool2d(kernel_size=2, stride=2)  # 주석 처리

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        #out = self.mp(out)  # 주석 처리
        return out

class SpecRNet(nn.Module):
    def __init__(self, d_args, **kwargs):
        super().__init__()

        self.device = kwargs.get("device", "cuda")

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(
            Residual_block2D(nb_filts=[1, 16], first=True)
        )
        self.block2 = nn.Sequential(Residual_block2D(nb_filts=[16, 32]))
        self.block4 = nn.Sequential(Residual_block2D(nb_filts=[32, 64]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc_attention0 = self._make_attention_fc(
            in_features=16, l_out_features=16
        )
        self.fc_attention2 = self._make_attention_fc(
            in_features=32, l_out_features=32
        )
        self.fc_attention4 = self._make_attention_fc(
            in_features=64, l_out_features=64
        )

        self.bn_before_gru = nn.BatchNorm2d(num_features=64)
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=d_args["gru_node"],
            num_layers=d_args["nb_gru_layer"],
            batch_first=True,
            bidirectional=True,
        )

        self.fc1_gru = nn.Linear(
            in_features=d_args["gru_node"] * 2, out_features=d_args["nb_fc_node"] * 2
        )

        self.fc2_gru = nn.Linear(
            in_features=d_args["nb_fc_node"] * 2,
            out_features=d_args["nb_classes"],
            bias=True,
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        # print("Shape after block0:", x0.shape)
        y0 = self.avgpool(x0).view(x0.size(0), -1)
        # print("Shape after avgpool (block0) and view:", y0.shape)
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), 1, 1)
        x = x0 * y0 + y0

        #print("Shape before maxpool (block0):", x.shape)
        #x = nn.MaxPool2d((2, 2))(x)  # 주석 처리
        #print("Shape after maxpool (block0):", x.shape)

        x2 = self.block2(x)
        # print("Shape after block2:", x2.shape)
        y2 = self.avgpool(x2).view(x2.size(0), -1)
        # print("Shape after avgpool (block2) and view:", y2.shape)
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), 1, 1)
        x = x2 * y2 + y2

        #print("Shape before maxpool (block2):", x.shape)
        #if x.size(2) > 1 and x.size(3) > 1:
        #    x = nn.MaxPool2d((2, 2))(x)  # 주석 처리
        #print("Shape after maxpool (block2):", x.shape)

        x4 = self.block4(x)
        # print("Shape after block4:", x4.shape)
        y4 = self.avgpool(x4).view(x4.size(0), -1)
        # print("Shape after avgpool (block4) and view:", y4.shape)
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), 1, 1)
        x = x4 * y4 + y4

        #print("Shape before maxpool (block4):", x.shape)
        #if x.size(2) > 1 and x.size(3) > 1:
        #    x = nn.MaxPool2d((2, 2))(x)  # 주석 처리
        #print("Shape after maxpool (block4):", x.shape)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.squeeze(-2)  # (batch_size, 64, height, width)
        # print("Shape after squeeze:", x.shape)
        x = x.flatten(start_dim=2)  # (batch_size, 64, height * width)
        # print("Shape after flatten:", x.shape)
        x = x.permute(0, 2, 1)  # (batch_size, height * width, 64)
        # print("Shape after permute:", x.shape)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # print("Shape after GRU:", x.shape)
        x = x[:, -1, :]
        # print("Shape after selecting last GRU output:", x.shape)
        x = self.fc1_gru(x)
        # print("Shape after fc1_gru:", x.shape)
        x = self.fc2_gru(x)
        # print("Shape after fc2_gru:", x.shape)

        return x

    def _make_attention_fc(self, in_features, l_out_features):
        l_fc = []
        l_fc.append(nn.Linear(in_features=in_features, out_features=l_out_features))
        return nn.Sequential(*l_fc)