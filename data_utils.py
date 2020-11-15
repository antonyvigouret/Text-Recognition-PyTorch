import string

import cv2
import numpy as np
import torch
from trdg.generators import GeneratorFromDict


class FakeTextImageGenerator:
    def __init__(self, batch_size=1, alphabet=string.printable):
        super(FakeTextImageGenerator).__init__()
        self.batch_size = batch_size
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.height = 32
        self.generator = GeneratorFromDict(
            length=5,
            allow_variable=True,
            language="en",
            size=32,
            background_type=1,
            fit=True,
            text_color="#000000,#888888",
        )

    def text_to_labels(self, text):
        """Translation of characters to unique integer values"""
        ret = []
        for char in text:
            ret.append(self.alphabet.find(char) + 1)
        ret.append(96)  # for seq2seq model (padding right with 96=EOS_idx)
        return ret

    def get_fake_image_gt(self):
        valid_text = False
        w = 1000
        while w > 500 or valid_text is False:
            im, text = next(self.generator)
            valid_text = True
            # for char in text:
            #     if char not in self.alphabet:
            #         valid_text = False
            #         continue
            im = np.array(im)
            w = im.shape[1]
        im = np.array(im)
        return im, text

    def pad_to_largest_image(self, images):
        sizes = [im.shape[1] for im in images]
        max_size = max(sizes)
        input_length = np.array([(max_size // 4 - 3) for im in images])
        images = [
            np.pad(im, ((0, 0), (0, max_size - im.shape[1]), (0, 0)), mode="edge") for im in images
        ]
        return images, input_length

    def pad_to_largest_label(self, labels):
        label_length = np.array([len(s) for s in labels])
        maxlen = max(len(s) for s in labels)
        labels_array = np.ones([self.batch_size, maxlen], dtype=int) * 97
        for i in range(self.batch_size):
            for j in range(len(labels[i])):
                labels_array[i][j] = int(labels[i][j])
        return labels_array, label_length

    def prepare_batch(self):
        images = []
        labels = []
        targets = []
        source_strings = []
        for _ in range(self.batch_size):
            w = 0
            while w < 10:
                im, text = self.get_fake_image_gt()
                w = im.shape[1]
            im = im / 255
            try:
                im = self.augment_image(im.astype("float32"))
            except:
                pass
            label = self.text_to_labels(text)
            for item in label:
                targets.append(item)
            images.append(im)
            labels.append(label)
            source_strings.append(text)
        images, input_length = self.pad_to_largest_image(images)
        labels, label_length = self.pad_to_largest_label(labels)
        self.batch = np.array(images)
        self.labels = labels
        self.input_length = input_length
        self.label_length = label_length
        self.targets = targets
        self.source_strings = source_strings

    def iter(self):
        # self.prepare_batch()
        while 1:
            self.prepare_batch()
            data = {
                "the_inputs": torch.from_numpy(self.batch).permute(0, 3, 1, 2),
                "the_labels": torch.from_numpy(self.labels),
                "input_length": torch.from_numpy(self.input_length),
                "label_length": torch.from_numpy(self.label_length),
                "targets": torch.LongTensor(self.targets),
            }
            yield data
