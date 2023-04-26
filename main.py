import argparse
import torch
import torch.nn as nn

from models.neuron2seq import *
from utils.util import *
from datasets.dataset import *
from engine import *

from path_util import *
from file_io import *


parser = argparse.ArgumentParser(
    description='Neuron Tracing Transformer')
# data specific
parser.add_argument('--data_file', default='/PBshare/SEU-ALLEN/Users/Gaoyu/neuronSegSR/Task501_neuron/data_splits.pkl',
                    type=str, help='dataset split file')
# training specific
# dataset
parser.add_argument('--seed', default=1025, type=int,
                    help='Random seed value')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--image_shape', default='32,64,64', type=str,
                    help='Input image shape')
parser.add_argument('--cpu', action="store_true",
                    help='Whether use gpu to train model, default True')
parser.add_argument('--num_classes', default=4, type=int,
                    help='the nums of classes')
parser.add_argument('--num_bins', default=64, type=int,
                    help='the nums of bins')
parser.add_argument('--node_len', default=3, type=int,
                    help='the nums of nodes in the sequence')
#model
parser.add_argument('--base_channels', default=24, type=int,
                    help="Output channels in the pre layer")
parser.add_argument('--down_kernel_list', default=[[1,3,3], [3,3,3], [3,3,3], [3,3,3]], type=list, nargs='+',
                    help="Down kernel list")
parser.add_argument('--stride_list', default=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], type=list, nargs='+',
                    help="Stride list")
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--heads', default=8, type=int,
                    help="Number of heads in the transformer")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.99, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--max_grad_norm', default=1.0, type=float,
                    help='Max gradient norm.')
parser.add_argument('--epochs', default=200, type=int,
                    help='maximal number of epochs')
parser.add_argument('--deterministic', action='store_true',
                    help='run in deterministic mode')
parser.add_argument('--val_frequency', default=20, type=int,
                    help='frequency of val')
parser.add_argument('--debug_frequency', default=5, type=int,
                    help='num of saving debug_image in one epoch')
parser.add_argument('--num_debug_save', default=5, type=int,
                    help='frequency of saving debug_image')
parser.add_argument('--checkpoint', default='', type=str,
                    help='Saved checkpoint')
parser.add_argument('--evaluation', action='store_true',
                    help='evaluation')
parser.add_argument('--phase', default='train')

# network specific
parser.add_argument('--save_folder', default='exps/temp',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if __name__ == '__main__':
    seed_everything(args.seed)
    args.image_shape = tuple(map(int, args.image_shape.split(',')))
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    
    tokenizer = Tokenizer(num_classes=args.num_classes, num_bins=args.num_bins, depth=args.image_shape[0],
                          height=args.image_shape[1], width=args.image_shape[2], max_len=args.node_len)
    args.pad_idx = tokenizer.PAD_code
    train_loader, _ = get_loaders(args, 'train', tokenizer)
    val_loader, _ = get_loaders(args, 'val', tokenizer)
    print(f'   {args}')

    model = Neu2seq(in_channels=1, base_channels=args.base_channels, encoder_depth=args.enc_layers, 
                    decoder_depth=args.dec_layers, down_kernel_list=args.down_kernel_list, stride_list=args.stride_list, dim=args.hidden_dim,
                    heads=args.heads, dropout=args.dropout, vocab_size=tokenizer.vocab_size, pad_idx=args.pad_idx)
    
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_training_steps = args.epochs * (len(train_loader.dataset) // args.batch_size)

    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps, num_training_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    
    train_eval(model, tokenizer, train_loader, val_loader, criterion, optimizer, lr_scheduler, step='batch', logger=None, args=args)
    
