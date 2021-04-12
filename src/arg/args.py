# -*- coding: UTF-8 -*-
import argparse
from src.utils.util import print_and_write_log, get_host_name
import platform
import os

parser = argparse.ArgumentParser(description='segmentation_tutorial', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--resume", action='store_true', default=True, help="resume model from pth file")

parser.add_argument("--dataparallel", action='store_true', default=False, help="use-ddp (default=False)")

# ---------------------- train/validate/test settings ----------------------
parser.add_argument("--n-epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate', )
parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay', )
parser.add_argument('--momentum', type=float, default=0.90, help='momentum', )

parser.add_argument("--train-batch-size", type=int, default=3, help="number of train batch size")
parser.add_argument("--val-batch-size", type=int, default=3, help="number of val batch size")

args = parser.parse_args()

args.raw_data_dir = os.path.expanduser(r'/ext1/lgsmile_dir/VOCdevkit/VOC2012')    

print_and_write_log(args)
