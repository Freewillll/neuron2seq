import numpy as np
import torch
import os, sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import skimage.morphology as morphology

from file_io import *


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
        #tokenized.append(self.EOS_code)

        return tokenized    
    
    def decode(self, tokens):
        """
        toekns: torch.LongTensor with shape [L]
        """
        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        token_list = tokens.tolist()
        if self.EOS_code in token_list:
            end = token_list.index(self.EOS_code)
            tokens = tokens[1:end]
        else:
            tokens = tokens[1:]
            #return None, None, False

        if len(tokens) % 4 != 0 or len(tokens) == 0:
            return None, None, False

        labels = []
        poses = []
        for i in range(3, len(tokens)+1, 4):
            label = tokens[i]
            bbox = tokens[i-3: i]
            labels.append(int(label))
            poses.append([int(item) for item in bbox])
        labels = np.array(labels) - self.num_bins
        poses = np.array(poses)
        poses = self.dequantize(poses)
        
        poses[:, 0] = poses[:, 0] * self.depth
        poses[:, 1] = poses[:, 1] * self.height
        poses[:, 2] = poses[:, 2] * self.width
        
        return labels, poses.astype('int32'), True
    
    @torch.no_grad()
    def visualization(self, img, token):
        img = np.repeat(img, 3, axis=0)
        img[:,:,:,:] = 0
        #img[0, :, :, :] = 0
        #img[2, :, :, :] = 0

        labels, poses, flag = self.decode(token)
        if flag == False:
            return None, False
        max_boundary = list(img[0].shape)
        max_boundary = [*map(lambda x: x - 1, max_boundary)]
        poses = np.clip(poses, 0, max_boundary)

        for idx, node in enumerate(poses):
            if labels[idx] == 0: # root
                img[:, node[0], node[1], node[2]] = [255,0,255]
            elif labels[idx] == 1: # branching point 
                img[:, node[0], node[1], node[2]] = [255,0,0]
            elif labels[idx] == 2: # tip
                img[:, node[0], node[1], node[2]] = [0,255,0]
            elif labels[idx] == 3: #boundary
                img[:, node[0], node[1], node[2]] = [0,0,255]

        selem = np.ones((1,1,3,3), dtype=np.uint8)
        img = morphology.dilation(img, selem)

        return img, True
