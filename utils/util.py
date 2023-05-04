import os,sys
import random
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import math
from transformers import top_k_top_p_filtering
import torch

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.image_util import unnormalize_normal
from path_util import *
from file_io import *


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    """Function to avoid numpy.random seed duplication across multi-threads"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def generate_square_subsequent_mask(sz, device, start_pos):
    mask = (torch.triu(torch.ones((sz, sz), device=device), start_pos)
            == 1)
    # mask = mask.float().masked_fill(mask == 1, float(
    #     '-inf')).masked_fill(mask == 0, float(0.0))
    return mask


def create_mask(tgt, pad_idx, seq_start_pos=5):
    """
    tgt: shape(N, L)
    """
    tgt_seq_len = tgt.shape[1]
    device = tgt.device

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device, seq_start_pos)
    tgt_padding_mask = (tgt == pad_idx)

    return tgt_mask, tgt_padding_mask


def read_marker(marker_file):
    poses = []
    labels = []
    with open(marker_file) as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            if line[0] == '#': continue
            x, y, z, *_, r, g, b = line.split(',')
            x = float(x)
            y = float(y)
            z = float(z)
            r = float(r)
            g = float(g)
            b = float(b)  
            type_ = 0
            if r:
                type_ = 1
            elif g:
                type_ = 2
            elif b:
                type_ = 3
            poses.append([z, y, x])
            labels.append(type_)
    
    return poses, labels


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.5f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
    

@torch.no_grad()
def save_image_debug(tokenizer, img_files, img, tokens, preds, epoch, phase, idx, args): 
    """
        Image Shape: b, c, z, y, x
        Tokens Shape: b, l
    """ 
    
    img_file = img_files[idx]
    prefix = get_file_prefix(img_file)

    img = (unnormalize_normal(img[idx].cpu().numpy())).astype(np.uint8)
    token = tokens[idx].clone().cpu().numpy()
    start = token[:5]
    print(f'lab: {token}')
    img_lab = tokenizer.visualization(img[:], token)
    
    if phase == 'train':
        out_lab_file = f'debug_epoch{epoch}_{prefix}_{phase}_lab.v3draw'
    else:
        out_lab_file = f'debug_epoch{epoch}_{prefix}_{phase}_lab.v3draw'
        
    save_image(os.path.join(args.save_folder, out_lab_file), img_lab)
        
    if preds != None:
        pred = torch.argmax(preds[idx], dim=-1).clone().cpu().numpy()
        pred = np.concatenate([start, pred], axis=0)
        print(f'pred: {pred}')
        img_pred = tokenizer.visualization(img, pred)

        if phase == 'train':
            out_pred_file = f'debug_epoch{epoch}_{prefix}_{phase}_pred.v3draw'
        else:
            out_pred_file = f'debug_epoch{epoch}_{prefix}_{phase}_pred.v3draw'

        save_image(os.path.join(args.save_folder, out_pred_file), img_pred)


@torch.no_grad()
def generate(model, img, seq, steps, top_k=0, top_p=1, args=None):
    img = img.to(args.device)
    seq = seq.to(args.device)

    if top_k != 0 or top_p != 1:
        sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)

    for i in range(steps):
        print(f'input: {seq}')
        preds = model.predict(img, seq, args)
        preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)
        preds = sample(preds)
        print(f'predict: {preds}')
        seq = torch.cat([seq, preds], dim=1)

    return seq.cpu()