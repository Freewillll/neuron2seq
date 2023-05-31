import argparse
import torch
import torch.nn as nn

from models.neuron2seq import *
from utils.util import *
from datasets.dataset import *
from engine import *


from path_util import *
from file_io import *

parser = argparse.ArgumentParser("Infer single image")
# data
parser.add_argument("--image", type=str, help="Path to image", default="./test_data/18869_8933_crop.v3draw")
parser.add_argument("--marker", type=str, help="Path to marker file", default="./test_data/18869_8933_crop.marker")
parser.add_argument('--checkpoint', default='./exps/exps015/debug/final.pth', type=str, help='Saved checkpoint')
parser.add_argument('--steps', default=15, type=int, help="Generate steps nums")

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
parser.add_argument('--image_shape', default='32,64,64', type=str,
                    help='Input image shape')
parser.add_argument('--num_classes', default=4, type=int,
                    help='Number of classes')
parser.add_argument('--num_bins', default=64, type=int,
                    help='Number of bins')
parser.add_argument('--node_len', default=3, type=int,
                    help='Number of nodes in the sequence')
parser.add_argument('--max_seq_len', default=20, type=int,
                    help='Length of the sequence')


parser.add_argument('--save_folder', default='exps/temp',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if __name__ == '__main__':
    args.image_shape = tuple(map(int, args.image_shape.split(','))) 
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(num_classes=args.num_classes, num_bins=args.num_bins, depth=args.image_shape[0],
                          height=args.image_shape[1], width=args.image_shape[2], max_len=args.node_len)
    args.pad_idx = tokenizer.PAD_code

    img_paths = [args.image]
    marker_paths = [args.marker]
    test_dataset = DatasetTest(img_paths, marker_paths, args.image_shape, tokenizer)

    model = Neu2seq_test(in_channels=1, base_channels=args.base_channels, encoder_depth=args.enc_layers, 
                decoder_depth=args.dec_layers, down_kernel_list=args.down_kernel_list, stride_list=args.stride_list, dim=args.hidden_dim,
                heads=args.heads, dropout=args.dropout, vocab_size=tokenizer.vocab_size, pad_idx=args.pad_idx)
    
    model.to(args.device)
    msg = model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    print(msg)
    model.eval()

    img, seq, _ = test_dataset[0]
    seq = seq[:5]
    img = img.unsqueeze(0)
    seq = seq.unsqueeze(0)
    print(f'seq: {seq}')

    img = img.to(args.device)
    seq = seq.to(args.device)
    
    # preds = model(img, seq)
    # preds = torch.argmax(preds, dim=-1)

    preds = generate(model, img, seq, 10, 0, 1, args)

    print(f'pred: {preds}')
    


