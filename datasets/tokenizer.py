import numpy as np
import torch
import os, sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


class Tokenizer:
    def __init__(self, num_classes: int, num_bins: int, depth: int, height: int, width: int, max_len=500):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.depth = depth
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

    def __call__(self, labels: list, poses: list, shuffle=True):
        assert len(labels) == len(poses), "labels and poses must have the same length"
        poses = np.array(poses)
        poses = np.clip(poses, 0, [self.depth-1, self.height-1, self.width-1])
        labels = np.array(labels)
        labels += self.num_bins
        labels = labels.astype('int')[:self.max_len]

        poses[:, 0] = poses[:, 0] / self.depth
        poses[:, 1] = poses[:, 1] / self.height
        poses[:, 2] = poses[:, 2] / self.width

        poses = self.quantize(poses)[:self.max_len]

        if shuffle:
            rand_idxs = np.arange(0, len(poses))
            np.random.shuffle(rand_idxs)
            labels = labels[rand_idxs]
            poses = poses[rand_idxs]

        tokenized = [self.BOS_code]
        for label, bbox in zip(labels, poses):
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
        assert len(tokens) % 4 == 0, "invalid tokens"

        labels = []
        poses = []
        for i in range(4, len(tokens)+1, 5):
            label = tokens[i]
            bbox = tokens[i-4: i]
            labels.append(int(label))
            poses.append([int(item) for item in bbox])
        labels = np.array(labels) - self.num_bins
        poses = np.array(poses)
        poses = self.dequantize(poses)
        
        poses[:, 0] = poses[:, 0] * self.depth
        poses[:, 1] = poses[:, 1] * self.height
        poses[:, 2] = poses[:, 2] * self.width
        
        return labels, poses