import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """Some Information about CRNN"""

    def __init__(self, nclass):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.mp2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.mp4 = nn.MaxPool2d((2, 1), stride=(2, 1))

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.mp6 = nn.MaxPool2d((2, 1), stride=(2, 1))

        self.conv7 = nn.Conv2d(512, 512, kernel_size=2)

        self.bidiLSTMs = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True)

        self.linear = nn.Linear(512, nclass)

    def forward(self, x):
        # [?, 32, W, 1] -> [?, 32, W, 64] -> [?, 16, W/2, 1]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.mp1(out)

        # [?, 16, W/2, 1] -> [?, 16, W/2, 128] -> [?, 8, W/4, 128]
        out = self.conv2(out)
        out = self.mp2(out)

        # [?, 8, W/4, 128] -> [?, 8, W/4, 256]
        out = self.conv3(out)

        # [?, 8, W/4, 256] -> [?, 8, W/2, 256] -> [?, 4, W/4, 256]
        out = self.conv4(out)
        out = self.mp4(out)

        # [?, 4, W/4, 512] -> [?, 4, W/4, 512]
        out = self.conv5(out)
        out = self.bn5(out)

        # [?, 4, W/4, 512] -> [?, 4, W/4, 512] -> [?, 2, W/4, 512]
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.mp6(out)

        # [?, 2, W/4, 512] -> [?, 1, W/4-3(?), 512]
        out = self.conv7(out)

        # [?, 1, W/4-3(?), 512]  -> [batch, depth_chanel, width_seq] = [?, W/4-3, 512]
        out = torch.squeeze(out, dim=2)
        # [seq, batch, chanel]
        out = out.permute(2, 0, 1)

        # [?, W/4-3, 512] -> # [?, W/4-3, 512] -> # [?, W/4-3, 512]
        out, _ = self.bidiLSTMs(out)

        # [?, W/4-3, 136(alphabet_size)]
        T, b, h = out.size()
        out = out.view(T * b, h)
        out = self.linear(out)
        out = out.view(T, b, -1)

        out = F.log_softmax(out, dim=2)
        return out
