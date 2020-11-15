import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Some Information about Encoder"""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size, hidden_size)

    def forward(self, input):
        output, (hidden, cell) = self.rnn(input)
        return output, hidden, cell


class Decoder(nn.Module):
    """Some Information about Decoder"""

    def __init__(self, hidden_size, output_size, emb_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(hidden_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        input = self.embedding(input)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        output = self.linear(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden, cell


class ConvSeq2Seq(nn.Module):
    """Some Information about ConvSeq2Seq"""

    def __init__(self, nclass):
        super(ConvSeq2Seq, self).__init__()
        self.nclass = nclass
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

        self.encoder = Encoder(512, 512)
        self.decoder = Decoder(512, nclass, nclass)

        self.linear = nn.Linear(512, nclass)

    def forward(self, x, target, teacher_forcing_ratio=1):
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

        out, hidden, cell = self.encoder(out)

        # tensor to store decoder outputs
        target_len = target.size()[1]
        batch_size = target.size()[0]
        outputs = torch.zeros(target_len, batch_size, self.nclass)

        input = torch.zeros_like(target[:, 0])

        for t in range(0, target_len):
            out, hidden, cell = self.decoder(input, hidden, cell)
            out = out.squeeze(0)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = out

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = out.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = target[:, t] if teacher_force else top1

        return outputs
