import string

import torch
from torch.nn import CTCLoss
import torch.optim as optim
from tqdm import tqdm

from crnn import CRNN
from data_utils import FakeTextImageGenerator
from utils import labels_to_text


def train(path=None):
    dataset = FakeTextImageGenerator(batch_size=32).iter()

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
        for i in tqdm(range(100)):
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
            loss = criterion(outputs, labels, input_length, label_length) / 32

            # print(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch: {epoch} | Loss: {running_loss/100}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss,
            },
            "checkpoint4.pt",
        )

    print("Finished Training")


if __name__ == "__main__":
    train()
