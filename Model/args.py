import argparse
import torch
# ordered by first character
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--output_dim', type=int, default=10,
                    help='classification category')
parser.add_argument('--epoch', type=int, default=100,
                    help='EPOCH')
parser.add_argument('--lr', type=int, default=0.0005,
                    help='learning rate')
'-----------info_enc-------------'
parser.add_argument('--factor', type=int, default=5)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_channel', type=int, default=128)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--e_layers', type=int, default=1)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=320)
parser.add_argument('--dropout', type=int, default=0.1)
'-------------------STN-------------------------'
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--STNembed_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--STN_heads', type=int, default=2)
parser.add_argument('--up_d', type=int, default=4)
parser.add_argument('--STNdropout', type=int, default=0.3)
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
'-------------------AvgPooling EEGNet-------------------------'
parser.add_argument('--EEG_channels', type=int, default=12)
parser.add_argument('--samples', type=int, default=151)
parser.add_argument('--dropoutRate', type=int, default=0.5)
parser.add_argument('--kernelLength', type=int, default=12)
parser.add_argument('--kernelLength2', type=int, default=16)
parser.add_argument('--F1', type=int, default=8)
parser.add_argument('--D', type=int, default=2)
parser.add_argument('--F2', type=int, default=16)

'----------------------------------------------------'
parser.add_argument('--gamma', type=float, default=0.6)

args = parser.parse_args()