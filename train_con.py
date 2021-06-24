from pathlib import Path
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
from time import perf_counter
import argparse
import os
import sys
import config
import losses as L
from data import SPDataset
import torch.utils.data as D
from torchtoolbox.tools import mixup_data, mixup_criterion
import model as M

from sklearn.metrics import confusion_matrix
from scheduler import GradualWarmupScheduler


def parse_args(argv=None):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-o', '--out', default=None)
    argparser.add_argument('-p', '--path', type=str, default='small_but_bigger_train')
    argparser.add_argument('-m', '--model', type=str, default='wrn')
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--img_size', type=int, default=256)
    argparser.add_argument('--train_ratio', type=float, default=0.8)
    argparser.add_argument('--loss', type=str, default='supcon')
    argparser.add_argument('--optim', type=str, default='SGD')
    argparser.add_argument('-wd', '--weight_decay', type=float, default=0)
    argparser.add_argument('--patience', type=int, default=5)
    argparser.add_argument('--log', action='store_true')
    argparser.add_argument('--new', action='store_true')
    argparser.add_argument('--options', type=str, default='r')
    argparser.add_argument('--classes', type=str, default=None)
    argparser.add_argument('--freeze', type=int, default=-1)
    argparser.add_argument('--warmup', type=int, default=0)
    argparser.add_argument('--lars', action='store_true')
    argparser.add_argument('--eval', type=str, default=None)
    argparser.add_argument('--filter_size', type=int, default=0)
    argparser.add_argument('--transfer_reg', type=float, default=0)
    argparser.add_argument('--transfer_decay', action='store_true')
    args = argparser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.log:
        log_f = config.log_path / '{}_{}'.format(args.path, args.train_ratio) / args.out
        log_f.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_f.with_suffix('.txt')
        sys.stdout = open(log_f, 'w')

    split_f = config.data_split_path / args.path / '{}.pt'.format(args.train_ratio)
    train_fn, val_fn = torch.load(split_f)

    train = SPDataset(args.path, train_fn, img_size=args.img_size, options=args.options, classes=args.classes, constrastive=True)
    val = SPDataset(args.path, val_fn, img_size=args.img_size, options=args.options, classes=args.classes, constrastive=True)
    batch_size = args.batch_size
    dtrain = D.DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=8)
    dval = D.DataLoader(val, batch_size=batch_size, num_workers=8)

    model = M.get_model(args.model, pretrained=not args.new)
    clf_name, clf_layer = M.get_clf_layer(model)
    new_clf_layer = M.Iden()
    M.set_layer(model, clf_name, new_clf_layer)

    if args.freeze >= 0:
        if args.model in ['wrn', 'rn', 'rnx']:
            for n, m in model.named_modules():
                if len(n) == 0 or len(n.split('.')) > 1:
                    continue
                if 'layer' in n:
                    layer = int(n[5:])
                    if layer > args.freeze:
                        break
                for param in m.parameters():
                    param.requires_grad = False

    model = M.SupCon(model, clf_layer.in_features)
    
    if args.transfer_reg > 0:
        transfer_loss_func = L.TranferLoss(model, args.transfer_reg, leave_last=len(list(new_clf_layer.parameters())), decay=args.transfer_decay)
        transfer_loss_func.cuda()

    args.loss = args.loss.lower()
    if args.loss == 'supcon':
        loss_func = L.SupConLoss()
    else:
        raise ValueError

    model = model.cuda()
    args.optim = args.optim.lower()
    if args.optim == 'adam':
        opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=args.weight_decay)
    if args.lars:
        from optimizers import LARC
        opt = LARC(opt)
    
    reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, verbose=True)
    if args.warmup > 0:
        warmup_lr = GradualWarmupScheduler(opt, multiplier=1, total_epoch=args.warmup)

    epochs = 100000
    best_val_loss = 1e8
    best_ep = 0
    # t = tqdm(total=len(dtrain))
    s_time = perf_counter()
    for ep in range(1, 1+epochs):

        loss_ep = 0
        model.train()
        # t.reset()
        pbar = tqdm(dtrain, disable= not os.environ.get("TQDM", False))
        for x, y in pbar:
            x = torch.cat([x[0], x[1]], dim=0)
            x, y = x.cuda(), y.cuda()
            features = model(x)
            f1, f2 = torch.chunk(features, 2, dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = loss_func(features, y)
            if args.transfer_reg > 0:
                loss += transfer_loss_func(model)

            opt.zero_grad()
            loss.backward()
            opt.step()
            if args.warmup > 0:
                warmup_lr.step()
            loss_ep += loss.item() * len(x)
            # t.update()
        loss_ep /= len(train)
        reduce_lr.step(loss_ep)
        print('Train {}: {}'.format(ep, loss_ep))
        
        y_pred, y_true = [], []
        model.eval()
        with torch.no_grad():
            loss_val = 0
            for x, y in dval:
                x = torch.cat([x[0], x[1]], dim=0)
                x, y = x.cuda(), y.cuda()
                features = model(x)
                f1, f2 = torch.chunk(features, 2, dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = loss_func(features, y)
                loss_val += loss.item() * len(x)
            loss_val /= len(val)
            print('Validation: {}'.format(loss_val))

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_ep = ep
            if args.out is not None:
                model_f = config.model_path / '{}_{}'.format(args.path, args.train_ratio) / args.out
                model_f.parent.mkdir(parents=True, exist_ok=True)
                model_f = model_f.with_suffix('.pt')
                torch.save({
                    'model': model.encoder.state_dict(),
                }, model_f)
        if ep - best_ep > args.patience:
            break

    print('Training time: {}'.format(perf_counter() - s_time))
    print('Best epoch: {}'.format(best_ep))
    print('Best validation loss: {}'.format(best_val_loss))
