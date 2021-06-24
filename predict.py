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
    argparser.add_argument('-m', '--model', type=str, default='efn_b5_ns')
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--img_size', type=int, default=256)
    argparser.add_argument('--csv_in', type=str, default='test.csv')
    argparser.add_argument('--use_words', type=str, default=None)

    args = argparser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = parse_args()

    with open(config.data_path / args.csv_in) as f:
        header = f.readline()
        test_fn = [line.split(',')[0] for line in f]
    train = SPDataset('train', img_size=args.img_size, use_words=args.use_words)
    test = SPDataset('test', ['test/{}'.format(fn) for fn in test_fn], args.img_size, test=True, use_words=args.use_words)

    num_classes = 42
    model = M.get_model(args.model)
    clf_name, clf_layer = M.get_clf_layer(model)
    new_clf_layer = nn.Linear(clf_layer.in_features, num_classes)
    M.set_layer(model, clf_name, new_clf_layer)
    if args.use_words:
        model = M.ImgText(model, len(test.words), num_classes)
    M.load_ckpt(model, config.model_path / args.inp)

    ngpu = torch.cuda.device_count()
    if ngpu > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(ngpu))).cuda()
    else:
        model.cuda()

    dtest = D.DataLoader(test, batch_size=args.batch_size, num_workers=8)
    dtrain = D.DataLoader(train, batch_size=args.batch_size, num_workers=8)
    model.eval()

    y_pred = []
    y_out = []
    y_true = []
    with torch.no_grad():
        for x, y in tqdm.tqdm(dtrain):
            x = util.cuda(x)
            out = model(x)
            y_out.append(out)
            y_true.append(y)
            y_pred.append(out.argmax(dim=-1))
    y_pred = torch.cat(y_pred).cpu()
    y_out = torch.cat(y_out).cpu()
    y_true = torch.cat(y_true).cpu()

    ana_f = config.ana_path / 'predict_train' / args.inp
    ana_f.parent.mkdir(parents=True, exist_ok=True)
    ana_f = ana_f.with_suffix('.pt')
    torch.save({
        'y_pred': y_pred,
        'y_out': y_out,
        'y_true': y_true,
        'filenames': train.filenames
    }, ana_f)

    y_pred = []
    y_out = []
    with torch.no_grad():
        for x in tqdm.tqdm(dtest):
            x = util.cuda(x)
            out = model(x)
            y_out.append(out)
            y_pred.append(out.argmax(dim=-1))
    y_pred = torch.cat(y_pred).cpu()
    y_out = torch.cat(y_out).cpu()

    ana_f = config.ana_path / 'predict_test' / args.inp
    ana_f.parent.mkdir(parents=True, exist_ok=True)
    ana_f = ana_f.with_suffix('.pt')
    torch.save({
        'y_pred': y_pred,
        'y_out': y_out,
        'filenames': test.filenames
    }, ana_f)

