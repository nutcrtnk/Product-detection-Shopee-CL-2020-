from data import SPDataset
import argparse
import torchvision.models as models
import torch.nn as nn
import torch
import config
import torch.utils.data as D
import model as M
from pathlib import Path
import tqdm
import util

def parse_args(argv=None):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--inp', type=str, required=True)
    argparser.add_argument('-p', '--path', type=str, default='test')
    argparser.add_argument('-m', '--model', type=str, default='efn_b5_ns')
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--img_size', type=int, default=256)
    argparser.add_argument('--csv_in', type=str, default='test.csv')
    argparser.add_argument('--csv_out', type=str, default=None)
    argparser.add_argument('--num_gpu', type=int, default=1)
    argparser.add_argument('--use_words', type=str, default=None)

    args = argparser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = parse_args()

    with open(config.data_path / args.csv_in) as f:
        header = f.readline()
        test_fn = [line.split(',')[0] for line in f]
    if args.csv_out is None:
        args.csv_out = 'test_{}.csv'.format(Path(args.inp).stem)
    print('Output directory:', config.data_path / args.csv_out)
    test = SPDataset(args.path, ['test/{}'.format(fn) for fn in test_fn], args.img_size, test=True, use_words=args.use_words)

    num_classes = 42
    model = M.get_model(args.model)
    clf_name, clf_layer = M.get_clf_layer(model)
    new_clf_layer = nn.Linear(clf_layer.in_features, num_classes)
    M.set_layer(model, clf_name, new_clf_layer)
    if args.use_words:
        model = M.ImgText(model, len(test.words), num_classes)
    M.load_ckpt(model, config.model_path / args.inp)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()

    dtest = D.DataLoader(test, batch_size=args.batch_size, num_workers=8)
    model.eval()
    y_pred = []
    with torch.no_grad():
        for x in tqdm.tqdm(dtest):
            x = util.cuda(x)
            y_pred.append(model(x).argmax(dim=-1))
    y_pred = torch.cat(y_pred).cpu()
    with open(config.data_path / args.csv_out, 'w') as f:
        f.write(header)
        for fn, y in zip(test_fn, y_pred):
            f.write('{},{:02d}\n'.format(fn,y))