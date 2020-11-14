import string

import cv2
import torch
from torch.nn import CTCLoss
import torch.optim as optim

from crnn import CRNN
from data_utils import FakeTextImageGenerator
from utils import decode_batch2


dataset = FakeTextImageGenerator(batch_size=4).iter()


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
            cv2.imshow("im", images[i].permute(1, 2, 0).numpy())
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    eval()
