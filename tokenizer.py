import numpy as np
import torch

class Tokenizer:
    def __init__(self, num_classes: int, num_bins: int, width: int, height: int, max_len=500):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.width = width
        self.height = height
        self.max_len = max_len

        self.BOS_code = num_classes + num_bins
        self.EOS_code = self.BOS_code + 1
        self.PAD_code = self.EOS_code + 1

        self.vocab_size = num_classes + num_bins + 3

    def quantize(self, x: np.array):
        """
        x is a real number in [0, 1]
        """
        return (x * (self.num_bins - 1)).astype('int')
    
    def dequantize(self, x: np.array):
        """
        x is an integer between [0, num_bins-1]
        """
        return x.astype('float32') / (self.num_bins - 1)

    def __call__(self, labels: list, bboxes: list, shuffle=True):
        assert len(labels) == len(bboxes), "labels and bboxes must have the same length"
        bboxes = np.array(bboxes)
        labels = np.array(labels)
        labels += self.num_bins
        labels = labels.astype('int')[:self.max_len]

        bboxes[:, 0] = bboxes[:, 0] / self.width
        bboxes[:, 2] = bboxes[:, 2] / self.width
        bboxes[:, 1] = bboxes[:, 1] / self.height
        bboxes[:, 3] = bboxes[:, 3] / self.height

        bboxes = self.quantize(bboxes)[:self.max_len]

        if shuffle:
            rand_idxs = np.arange(0, len(bboxes))
            np.random.shuffle(rand_idxs)
            labels = labels[rand_idxs]
            bboxes = bboxes[rand_idxs]

        tokenized = [self.BOS_code]
        for label, bbox in zip(labels, bboxes):
            tokens = list(bbox)
            tokens.append(label)

            tokenized.extend(list(map(int, tokens)))
        tokenized.append(self.EOS_code)

        return tokenized    
    
    def decode(self, tokens: torch.tensor):
        """
        toekns: torch.LongTensor with shape [L]
        """
        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        tokens = tokens[1:-1]
        assert len(tokens) % 5 == 0, "invalid tokens"

        labels = []
        bboxes = []
        for i in range(4, len(tokens)+1, 5):
            label = tokens[i]
            bbox = tokens[i-4: i]
            labels.append(int(label))
            bboxes.append([int(item) for item in bbox])
        labels = np.array(labels) - self.num_bins
        bboxes = np.array(bboxes)
        bboxes = self.dequantize(bboxes)
        
        bboxes[:, 0] = bboxes[:, 0] * self.width
        bboxes[:, 2] = bboxes[:, 2] * self.width
        bboxes[:, 1] = bboxes[:, 1] * self.height
        bboxes[:, 3] = bboxes[:, 3] * self.height
        
        return labels, bboxes