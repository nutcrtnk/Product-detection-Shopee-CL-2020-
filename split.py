from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import config
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('-p', '--path', type=str, default='small_but_bigger_train')
argparser.add_argument('-t', '--train_ratio', type=float, default=0.8)
args = argparser.parse_args()

filenames = []
p = config.data_path / args.path
for pic_f in p.glob(r'*/*.jpg'):
    filename = '{}/{}'.format(pic_f.parent.name, pic_f.name)
    filenames.append(filename)
train_fn, val_fn = train_test_split(filenames, train_size=args.train_ratio)
save_fn =config.data_split_path / args.path / '{}.pt'.format(args.train_ratio)
save_fn.parent.mkdir(parents=True, exist_ok=True)
torch.save([train_fn, val_fn], save_fn)