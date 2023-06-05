
import numpy as np
import pickle
import torch.utils.data as tudata
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import SimpleITK as sitk
import torch
import sys
import os
from torch.utils.data import RandomSampler

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swc_handler import parse_swc, write_swc
from datasets.augmentation import InstanceAugmentation
from datasets.treeparser import trim_out_of_box, swc_to_seq
from utils.util import *

# To avoid the recursionlimit error
sys.setrecursionlimit(30000)

class GenericDataset(tudata.Dataset):

    def __init__(self, split_file, tokenizer = None, phase='train', imgshape=(32, 64, 64)):
        self.data_list = self.load_data_list(split_file, phase)
        self.imgshape = imgshape
        self.tokenizer = tokenizer
        print(f'Image shape of {phase}: {imgshape}')
        self.phase = phase
        self.augment = InstanceAugmentation(p=0.2, imgshape=imgshape, phase=phase)

    @staticmethod
    def load_data_list(split_file, phase):

        with open(split_file, 'rb') as fp:
            data_dict = pickle.load(fp)

        if phase == 'train' or phase == 'val':
            return data_dict[phase]

        elif phase == 'test':
            dd = data_dict['test']
            return dd
        else:
            raise ValueError

    def __getitem__(self, index):
        img, seqs, imgfile, swcfile = self.pull_item(index)
        return img, seqs, imgfile, swcfile

    def __len__(self):
        return len(self.data_list)

    def pull_item(self, index):
        imgfile, swcfile = self.data_list[index]
        # parse, image should in [c,z,y,x] format
        img = np.load(imgfile)['data']
        if img.ndim == 3:
            img = img[None]
        tree = None
        if swcfile is not None and self.phase != 'test':
            tree = parse_swc(swcfile)

        # random augmentation
        img, tree = self.augment(img, tree)

        if tree is not None and self.phase != 'test':
            tree_crop = trim_out_of_box(tree, img[0].shape, True)
            poses = []
            labels = []
            if len(tree_crop) != 0:
                poses, labels = swc_to_seq(tree_crop, img[0].shape, max_level=1)
                seqs = self.tokenizer(labels, poses, False)
                seqs = torch.LongTensor(seqs)
                return torch.from_numpy(img.astype(np.float32)), seqs, imgfile, swcfile

        else:
            poses = np.random.randn(3)
            labels = np.random.randn(1) > 0.5
            target = {'poses': torch.tensor(poses, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}
            return torch.from_numpy(img.astype(np.float32)), target, imgfile, swcfile


def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length
    """
    image_batch, seq_batch, imgfile_batch, swcfile_batch = [], [], [], []
    for image, seq, imgfile, swcfile in batch:
        image_batch.append(image)
        seq_batch.append(seq)
        imgfile_batch.append(imgfile)
        swcfile_batch.append(swcfile)

    seq_batch = pad_sequence(
        seq_batch, padding_value=pad_idx, batch_first=True)
    for i in range(seq_batch.size(0)):
        idx = (seq_batch[i] == pad_idx).nonzero(as_tuple=True)[0]
        if len(idx) != 0:
            idx = idx[0]
            seq_batch[i][idx] = pad_idx - 1

    if max_len:
        pad = torch.ones(seq_batch.size(0), max_len -
                         seq_batch.size(1)).fill_(pad_idx).long()
        seq_batch = torch.cat([seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    return image_batch, seq_batch, imgfile_batch, swcfile_batch


def get_loaders(args, phase, tokenizer):
    dset = GenericDataset(args.data_file, tokenizer, phase=phase, imgshape=args.image_shape)
    # sampler = RandomSampler(dset)
    loader = tudata.DataLoader(dset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True, pin_memory=True,
                               drop_last=True,
                               collate_fn=partial(collate_fn, max_len=0, pad_idx=tokenizer.PAD_code),
                               worker_init_fn=worker_init_fn)
    dset_iter = iter(loader)
    return loader, dset_iter


class DatasetTest(tudata.Dataset):
    def __init__(self, img_paths, marker_paths, imgshape, tokenizer):
        self.img_paths = img_paths
        self.marker_paths = marker_paths
        self.tokenizer = tokenizer
        self.augment = InstanceAugmentation(p=0.2, imgshape=imgshape, phase='test')

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = os.path.split(img_path)[-1]
        marker_path = self.marker_paths[idx]
        img = load_image(img_path)
        poses, labels = read_marker(marker_path)
        seqs = self.tokenizer(labels, poses, False)
        seqs = torch.LongTensor(seqs)
        return torch.from_numpy(img.astype(np.float32)), seqs, img_name

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':

    import skimage.morphology as morphology
    # import cv2 as cv
    from torch.utils.data.distributed import DistributedSampler
    import utils.util as util
    # import matplotlib.pyplot as plt
    from utils.image_util import *
    from datasets.tokenizer import Tokenizer
    # from datasets.mip import *
    # from train import *

    split_file = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl'
    idx = 1
    imgshape = (32, 64, 64)

    tokenizer = Tokenizer(num_classes=4, num_bins=64, depth=32,
                          width=64, height=64, max_len=3)

    dataset = GenericDataset(split_file, tokenizer, 'train', imgshape=imgshape)

    loader = tudata.DataLoader(dataset, 4, 
                                num_workers=4, 
                                shuffle=False, pin_memory=True,
                                drop_last=True, 
                                collate_fn=partial(collate_fn, max_len=0, pad_idx=tokenizer.PAD_code),
                                worker_init_fn=util.worker_init_fn)

    for i, batch in enumerate(loader):
        img, seqs, imgfile, swcfile = batch
        print(seqs)
        break
        # print(targets.shape)
        # save_image_in_training(imgfiles, img, seq, cls_, pred=None, phase='train', epoch=1, idx=0)

