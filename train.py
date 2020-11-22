import string

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import CTCLoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from cnn_seq2seq import ConvSeq2Seq
from cnn_seq2seq import Decoder
from cnn_seq2seq import Encoder
from cnn_seq2seq_att import ConvSeq2SeqAtt
from crnn import CRNN
from data_utils import FakeTextImageGenerator
from utils import labels_to_text
from utils import text_to_labels


def train(path=None):
    dataset = FakeTextImageGenerator(batch_size=16).iter()

    criterion = CTCLoss(reduction="mean", zero_infinity=True)

    net = CRNN(nclass=100).float()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    if path:
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"model current epoch: {epoch} with loss: {loss}")

    # loop over the dataset multiple times
    for epoch in range(1, 1000):
        running_loss = 0.0
        loop = tqdm(range(100))
        for i in loop:
            data = next(dataset)
            images = data["the_inputs"]
            labels = data["the_labels"]
            input_length = data["input_length"]
            label_length = data["label_length"]
            targets = data["targets"]

            # print("target", targets)
            # print("target l", targets.size())
            # print("label_l", label_length)
            # print("label_l l", label_length.size())
            # print("pred_l", input_length)
            # print("pred_l l", input_length.size())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images.float())
            # print(outputs[8, 0, :])
            # print(outputs[:, 0, :])
            # print(outputs.size())
            loss = criterion(outputs, labels, input_length, label_length)

            # print(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(epoch=epoch, loss=(running_loss / (i + 1)))

        # print(f"Epoch: {epoch} | Loss: {running_loss/100}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss,
            },
            "checkpoint5.pt",
        )

    print("Finished Training")


def train_cs2s(path=None):
    alphabet = string.printable
    nclass = len(alphabet)

    writer = SummaryWriter()
    dataset = FakeTextImageGenerator(batch_size=4).iter()

    criterion = CrossEntropyLoss(ignore_index=97)

    encoder = Encoder(512, 512, 1, 0)
    decoder = Decoder(512, 100, 100, 1, 0)
    net = ConvSeq2Seq(encoder, decoder, nclass=nclass).float()

    optimizer = optim.Adam(net.parameters(), lr=0.003)

    if path:
        net2 = CRNN(nclass=100).float()
        checkpoint = torch.load(path)
        net2.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # epoch = checkpoint["epoch"]
        # loss = checkpoint["loss"]
        # print(f"model current epoch: {epoch} with loss: {loss}")
        print(net2)

        net.conv1.load_state_dict(net2.conv1.state_dict())
        net.conv2.load_state_dict(net2.conv2.state_dict())
        net.conv3.load_state_dict(net2.conv3.state_dict())
        net.conv4.load_state_dict(net2.conv4.state_dict())
        net.conv5.load_state_dict(net2.conv5.state_dict())
        net.conv6.load_state_dict(net2.conv6.state_dict())
        net.conv7.load_state_dict(net2.conv7.state_dict())
    net.train()

    # loop over the dataset multiple times
    step = 0
    for epoch in range(1, 1000):
        running_loss = 0.0
        loop = tqdm(range(100))
        for i in loop:
            data = next(dataset)
            images = data["the_inputs"]
            labels = data["the_labels"]
            input_length = data["input_length"]
            label_length = data["label_length"]
            targets = data["targets"]

            # print("target", targets)
            # print("target l", targets.size())
            # print("label_l", label_length)
            # print("label_l l", label_length.size())
            # print("pred_l", input_length)
            # print("pred_l l", input_length.size())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images.float(), labels, 0.5)
            # permute batchsize and seq_len dim to match labels when using .view(-1, output.size()[2])
            outputs = outputs.permute(1, 0, 2)
            # print(outputs[8, 0, :])
            # print(outputs[:, 0, :])
            # print(outputs.size())
            # print(labels.size())
            output_argmax = outputs.argmax(2)
            # print(output_argmax.view(-1))
            # print(labels.reshape(-1))
            loss = criterion(outputs.reshape(-1, 100), labels.reshape(-1))

            writer.add_scalar("loss", loss.item(), step)
            step += 1
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(epoch=epoch, Loss=(running_loss / (i + 1)))

        # print(f"Epoch: {epoch} | Loss: {running_loss/100}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss,
            },
            "cs2s_good.pt",
        )
        torch.save(net, "model_test_pretrained.pt")

    print("Finished Training")


def train_cs2satt(path=None):
    writer = SummaryWriter()
    dataset = FakeTextImageGenerator(batch_size=8).iter()

    criterion = CrossEntropyLoss(ignore_index=97)

    net = ConvSeq2SeqAtt(nclass=100).float()

    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    if path:
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"model current epoch: {epoch} with loss: {loss}")
    net.train()

    # loop over the dataset multiple times
    step = 0
    for epoch in range(1, 1000):
        running_loss = 0.0
        loop = tqdm(range(100))
        for i in loop:
            data = next(dataset)
            images = data["the_inputs"]
            labels = data["the_labels"]
            input_length = data["input_length"]
            label_length = data["label_length"]
            targets = data["targets"]

            # print("target", targets)
            # print("target l", targets.size())
            # print("label_l", label_length)
            # print("label_l l", label_length.size())
            # print("pred_l", input_length)
            # print("pred_l l", input_length.size())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images.float(), labels, 0.5)
            # permute batchsize and seq_len dim to match labels when using .view(-1, output.size()[2])
            outputs = outputs.permute(1, 0, 2)
            # print(outputs[8, 0, :])
            # print(outputs[:, 0, :])
            # print(outputs.size())
            # print(labels.size())
            output_argmax = outputs.argmax(2)
            # print(output_argmax.view(-1))
            # print(labels.reshape(-1))
            loss = criterion(outputs.reshape(-1, 100), labels.reshape(-1))

            # print(loss.item())
            writer.add_scalar("loss", loss.item(), step)
            step += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(epoch=epoch, Loss=(running_loss / (i + 1)))

        print(f"Epoch: {epoch} | Loss: {running_loss/100}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss,
            },
            "cs2satt_good.pt",
        )
        # torch.save(net, "model_test_pretrained.pt")

    print("Finished Training")


if __name__ == "__main__":
    train_cs2satt("cs2satt_good.pt")
