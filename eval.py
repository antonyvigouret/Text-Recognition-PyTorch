import string

import cv2
import torch
from torch.nn import CTCLoss
import torch.optim as optim

from cnn_seq2seq import ConvSeq2Seq
from cnn_seq2seq import Decoder
from cnn_seq2seq import Encoder
from cnn_seq2seq_att import ConvSeq2SeqAtt
from crnn import CRNN
from data_utils import FakeTextImageGenerator
from utils import decode_batch2
from utils import labels_to_text


dataset = FakeTextImageGenerator(batch_size=6).iter()


def eval(path="checkpoint3.pt"):
    net = CRNN(nclass=100).double()
    optimizer = optim.Adam(net.parameters())

    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"model current epoch: {epoch} with loss: {loss}")

    net.eval()

    while 1:
        data = next(dataset)
        images = data["the_inputs"]
        labels = data["the_labels"]
        input_length = data["input_length"]
        label_length = data["label_length"]

        preds = net(images).detach()
        pred_texts, probs = decode_batch2(preds, string.printable)
        for i in range(len(pred_texts)):
            print(pred_texts[i], probs[i])
            print(images[i].size())
            # cv2.imshow("im", images[i].permute(1, 2, 0).numpy())
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


def eval_cs2s(path="cs2s_good.pt"):
    encoder = Encoder(512, 512, 1, 0)
    decoder = Decoder(512, 100, 100, 1, 0)
    net = ConvSeq2Seq(encoder, decoder, nclass=100).float()
    optimizer = optim.Adam(net.parameters())

    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"model current epoch: {epoch} with loss: {loss}")

    net.eval()

    with torch.no_grad():
        while 1:
            data = next(dataset)
            images = data["the_inputs"]
            labels = data["the_labels"]
            input_length = data["input_length"]
            label_length = data["label_length"]

            preds = net(images.float(), labels, 0).detach().permute(1, 0, 2)
            for i in range(len(preds)):
                print("labels", labels[i])
                print("preds", preds[i].argmax(1))
                print(labels_to_text(preds[i, :, :].argmax(1), string.printable))
                cv2.imshow("im", images[i].permute(1, 2, 0).numpy())
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def eval_cs2satt(path="cs2satt_good.pt"):
    net = ConvSeq2SeqAtt(nclass=100).float()
    optimizer = optim.Adam(net.parameters())

    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"model current epoch: {epoch} with loss: {loss}")

    net.eval()

    with torch.no_grad():
        while 1:
            data = next(dataset)
            images = data["the_inputs"]
            labels = data["the_labels"]
            input_length = data["input_length"]
            label_length = data["label_length"]

            preds = net(images.float(), labels, 0).detach().permute(1, 0, 2)
            for i in range(len(preds)):
                print("labels", labels[i])
                print("preds", preds[i].argmax(1))
                print(labels_to_text(preds[i, :, :].argmax(1), string.printable))
                cv2.imshow("im", images[i].permute(1, 2, 0).numpy())
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    eval_cs2satt()
