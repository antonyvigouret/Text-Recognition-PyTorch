import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Some Information about Encoder"""

    def __init__(self, input_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=p, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input):
        encoder_states, (hidden, cell) = self.rnn(input)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        return encoder_states, hidden, cell


class AttentionDecoder(nn.Module):
    """Some Information about Decoder"""

    def __init__(self, hidden_size, output_size, emb_dim, num_layers, p):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(100, emb_dim)
        self.rnn = nn.LSTM(hidden_size * 2 + emb_dim, hidden_size, num_layers, dropout=p)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, encoder_states, hidden, cell):
        input = input.unsqueeze(0)
        # input shape: (1, N, 1)
        input = self.embedding(input)
        # input shape: (1, N, emb_size)

        seq_len = encoder_states.shape[0]
        h_reshaped = hidden.repeat(seq_len, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        # attention = attention.permute(1, 2, 0)
        # encoder_states = encoder_states.permute(1, 0, 2)
        # context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)

        rnn_input = torch.cat((context_vector, input), dim=2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output shape: (1, N, output_size=nclass)
        output = self.linear(output).squeeze(0)
        # output = F.log_softmax(output, dim=2) dont need because of nn.CrossEntropy
        return output, hidden, cell


class ConvSeq2SeqAtt(nn.Module):
    """Some Information about ConvSeq2Seq"""

    def __init__(self, nclass):
        super(ConvSeq2SeqAtt, self).__init__()
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

        self.encoder = Encoder(512, 1024, 1, 0)
        self.decoder = AttentionDecoder(1024, nclass, nclass, 1, 0)

    def forward(self, x, target, teacher_forcing_ratio=1):
        #  (N, 3, 32, W) -> (N, 64, 16, W/2)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.mp1(out)

        # (N, 64, 16, W/2) -> (N, 128, 8, W/4)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.mp2(out)

        # (N, 128, 8, W/4) -> (N, 256, 8, W/4)
        out = self.conv3(out)
        out = self.relu(out)

        # (N, 256, 8, W/4) -> (N, 256, 4, W/4)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.mp4(out)

        # (N, 256, 4, W/4) -> (N, 512, 4, W/4)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        # (N, 512, 4, W/4) -> (N, 512, 2, W/4)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)
        out = self.mp6(out)

        # (N, 512, 2, W/4) ->  (N, 512, 1, W/4-)
        out = self.conv7(out)
        out = self.relu(out)

        # (N, 512, 1, W/4-)  -> (N, 512, W/4-)
        out = torch.squeeze(out, dim=2)
        # (t, n, 512)
        out = out.permute(2, 0, 1)

        encoder_states, hidden, cell = self.encoder(out)

        # tensor to store decoder outputs
        target_len = target.size(1)
        batch_size = target.size(0)
        outputs = torch.zeros(target_len, batch_size, self.nclass)

        input = torch.zeros_like(target[:, 0])  # start token = 0
        # input shape: (N, 1)

        for t in range(0, target_len):
            # print(hidden)
            out, hidden, cell = self.decoder(input, encoder_states, hidden, cell)
            # out shape: (N, nclass)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = out

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = out.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            # target shape: (N, len_target)
            input = target[:, t] if teacher_force else top1

        # output shape: (seq_len, N, nclass)
        return outputs
