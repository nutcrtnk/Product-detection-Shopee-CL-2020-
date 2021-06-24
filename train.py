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
import model as M
import util
import copy

from sklearn.metrics import confusion_matrix
from scheduler import GradualWarmupScheduler


def parse_args(argv=None):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-o', '--out', default=None)
    argparser.add_argument('-p', '--path', type=str,
                           default='small_but_bigger_train')
    argparser.add_argument('-m', '--model', type=str, default='efn_b5_ns')
    argparser.add_argument('--batch_size', type=int,
                           default=32)  # 16 for efn b7 600
    argparser.add_argument('--img_size', type=int, default=256) # 456 for b5  600 for b7
    argparser.add_argument('--train_ratio', type=float, default=0.8)
    argparser.add_argument('--loss', type=str, default='ce')
    argparser.add_argument('--gamma', type=float, default=2.)
    argparser.add_argument('--optim', type=str, default='SGD')
    argparser.add_argument('--base_lr', type=float, default=0.004)

    argparser.add_argument('-wd', '--weight_decay', type=float, default=0)
    argparser.add_argument('--patience', type=float, default=2)
    argparser.add_argument('--log', action='store_true')
    argparser.add_argument('--new', action='store_true')
    argparser.add_argument('--options', type=str, default='r')
    argparser.add_argument('--ovr', type=int, default=None)
    argparser.add_argument('--classes', type=str, default=None)
    argparser.add_argument('--val_classes', type=str, default=None)
    argparser.add_argument('--freeze', type=int, default=-1)
    argparser.add_argument('--freeze_enc', action='store_true')
    argparser.add_argument('--warmup', type=int, default=0)
    argparser.add_argument('--not-save', action='store_true')
    argparser.add_argument('--lars', action='store_true')
    argparser.add_argument('--zero_init', action='store_true')
    argparser.add_argument('--mixup', action='store_true')
    argparser.add_argument('--filter_size', type=int, default=0)
    argparser.add_argument('--transfer_reg', type=float, default=0)
    argparser.add_argument('--transfer_decay', action='store_true')

    argparser.add_argument('--full_epoch', type=int, default=0)
    argparser.add_argument('--load_con', type=str, default=None)
    argparser.add_argument('--load_ckpt', type=str, default=None)
    argparser.add_argument('--eval', action='store_true')
    argparser.add_argument('--num_gpu', type=int, default=1)
    argparser.add_argument('--extra', type=str, default=None)
    argparser.add_argument('--use_words', type=str, default=None)

    args = argparser.parse_args(argv)
    return args


def data_generator(dataloader):
    while True:
        for data in dataloader:
            yield data


class Optimize:

    def __init__(self, parameters, args):
        args.optim = args.optim.lower()
        scaled_lr = args.base_lr * (args.batch_size / 32)
        if args.optim == 'adam':
            opt = optim.Adam(parameters, lr=scaled_lr,
                            weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            opt = optim.SGD(parameters, lr=scaled_lr,
                            momentum=0.9, weight_decay=args.weight_decay)
        if args.lars:
            from optimizers import LARC
            opt = LARC(opt)

        self.reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, verbose=True)
        if args.warmup > 0:
            self.warmup_lr = GradualWarmupScheduler(
                opt, multiplier=1, total_epoch=args.warmup)
        else:
            self.warmup_lr = None
        self.opt = opt

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()
        if self.warmup_lr is not None:
            self.warmup_lr.step()


def objective_function(model, args):
    args.loss = args.loss.lower()
    if args.ovr is not None:
        loss_func = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        loss_func = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        loss_func = L.FocalLoss(num_classes=num_classes, gamma=args.gamma)
    elif args.loss == 'sce':
        loss_func = L.SCELoss(num_classes=num_classes)
    elif args.loss == 'mae':
        loss_func = L.MAELoss(num_classes=num_classes)
    elif args.loss == 'lq':
        loss_func = L.LQLoss(num_classes=num_classes)
    elif args.loss == 'vat':
        loss_func = L.VATLoss()
    else:
        raise ValueError

    def step_func(x, y):
        if args.ovr is not None:
            y = (y == args.ovr).float()
            loss = loss_func(model(x).flatten(),
                                torch.abs(y - 0.01))
        elif args.mixup:
            x, labels_a, labels_b, lam = L.mixup_data(x, y, 0.2)
            loss = nn.CrossEntropyLoss()
            loss = L.mixup_criterion(
                loss, model(x), labels_a, labels_b, lam)
        elif args.loss == 'vat':
            loss = loss_func(model, x, y)
        else:
            loss = loss_func(model(x), y)
        return loss
    return step_func


if __name__ == '__main__':
    args = parse_args()
    if args.full_epoch > 0:
        assert args.out
        folder = args.path
    else:
        folder = '{}_{}'.format(args.path, args.train_ratio)

    if args.log:
        log_f = config.log_path / folder / args.out
        log_f.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_f.with_suffix('.txt')
        sys.stdout = open(log_f, 'w')

    if args.ovr:
        num_classes = 1
    elif args.classes:
        args.classes = [int(c) for c in args.classes.split(',')]
        num_classes = len(args.classes)
    else:
        num_classes = 42
    
    if args.val_classes:
        args.val_classes = [int(c) for c in args.val_classes.split(',')]

    batch_size = args.batch_size
    if args.full_epoch <= 0:

        split_f = config.data_split_path / \
            args.path / '{}.pt'.format(args.train_ratio)
        train_fn, val_fn = torch.load(split_f)
        val = SPDataset(args.path, val_fn,
                img_size=args.img_size, classes=args.val_classes if args.val_classes else args.classes, use_words=args.use_words)
        dval = D.DataLoader(val, batch_size=batch_size, num_workers=8)
    else:
        train_fn ,val_fn = None, None
        val, dval = None, None
    
    train = SPDataset(args.path, train_fn, img_size=args.img_size,
                        options=args.options, classes=args.classes,  filter_size=args.filter_size, use_words=args.use_words)
    
    model = M.get_model(args.model, pretrained=not args.new)
    clf_name, clf_layer = M.get_clf_layer(model)
    if args.load_con:
        M.set_layer(model, clf_name, M.Iden())
        M.load_ckpt(model, config.model_path / folder / args.load_con)
    new_clf_layer = nn.Linear(clf_layer.in_features, num_classes)
    M.set_layer(model, clf_name, new_clf_layer)

    if args.use_words:
        model = M.ImgText(model, len(train.words), num_classes)
        img_model = model.img_encoder
    else:
        img_model = model

    if args.load_ckpt:
        M.load_ckpt(model, config.model_path / folder / args.load_ckpt)
    elif args.zero_init:
        [nn.init.zeros_(p) for p in new_clf_layer.parameters()]

    if args.freeze_enc:
        for p in img_model.parameters():
            p.requires_grad = False
        for p in new_clf_layer.parameters():
            p.requires_grad = True

    if args.freeze >= 0:
        if args.model in ['wrn', 'rn', 'rnx']:
            for n, m in img_model.named_modules():
                if len(n) == 0 or len(n.split('.')) > 1:
                    continue
                if 'layer' in n:
                    layer = int(n[5:])
                    if layer > args.freeze:
                        break
                for param in m.parameters():
                    param.requires_grad = False

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    epochs = 100000 if args.full_epoch <= 0 else args.full_epoch
    best_acc = 0
    best_ep = 0
    loss_ep = util.AverageMeter()
    s_time = perf_counter()
    ep = 0
    eval_step = min(10000, len(train))

    dtrain = D.DataLoader(train, shuffle=True,
                          batch_size=batch_size, num_workers=8)
    train_gen = data_generator(dtrain)
    obj_func = objective_function(model, args)

    model_ext = None
    params = model.parameters()
    if args.extra:
        assert args.transfer_reg > 0
        train_ext = SPDataset(args.extra, img_size=args.img_size,
                        options=args.options, ymap='auto', filter_size=args.filter_size)
        dtrain_ext = D.DataLoader(train_ext, shuffle=True,
                            batch_size=batch_size, num_workers=8)
        train_gen_ext = data_generator(dtrain_ext)

        model_ext = copy.deepcopy(img_model)
        new_ext_clf_layer = nn.Linear(clf_layer.in_features, len(train_ext.labels))
        M.set_layer(model_ext, clf_name, new_ext_clf_layer)
        if args.num_gpu > 1:
            model_ext = torch.nn.DataParallel(
                model_ext, device_ids=list(range(args.num_gpu))).cuda()
        else:
            model_ext = model_ext.cuda()
        obj_func_ext = objective_function(model_ext, args)
        transfer_loss_func = L.TransferLoss(args.transfer_reg, decay=args.transfer_decay)
        params = list(params) + list(model_ext.parameters())

    elif args.transfer_reg > 0:
        transfer_loss_func = L.PretrainedTransferLoss(img_model, args.transfer_reg, leave_last=len(
            list(new_clf_layer.parameters())), decay=args.transfer_decay)
        transfer_loss_func.cuda()
    

    opt = Optimize(params, args)

    while ep < epochs:

        if not args.eval:
            pbar = tqdm(total=eval_step, disable=not os.environ.get("TQDM", False))
            loss_ep.reset()

            while loss_ep.count < eval_step:
                model.train()
                x, y = next(train_gen)
                x, y = util.cuda(x), y.cuda()
                loss = obj_func(x, y)
                if args.transfer_reg > 0:
                    if model_ext:
                        x, y = next(train_gen_ext)
                        x, y = x.cuda(), y.cuda()
                        loss += obj_func_ext(x, y)
                    loss += transfer_loss_func(img_model, model_ext)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_ep.update(loss.item(), len(y))
                pbar.update(len(y))
            pbar.close()
            opt.reduce_lr.step(loss_ep.avg)
            ep += loss_ep.count / len(train)
            print('Train {}: {}'.format(ep, loss_ep.avg))

        if args.full_epoch <= 0:
            
            y_pred, y_true = [], []
            model.eval()
            with torch.no_grad():
                for x, y in dval:
                    x, y = util.cuda(x), y.cuda()
                    if args.ovr is not None:
                        y = (y == args.ovr).long()
                        y_pred.append((model(x) > 0.5).flatten().long())
                    else:
                        if args.val_classes:
                            out = model(x)[:, args.val_classes]
                            y_pred.append(out.argmax(dim=-1))
                        else:
                            y_pred.append(model(x).argmax(dim=-1))
                    y_true.append(y)
            y_pred = torch.cat(y_pred).cpu()
            y_true = torch.cat(y_true).cpu()

            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')
            acc = (y_pred == y_true).sum().float() / len(y_true)
            print('acc: {:.4f}, fma: {:.4f}, fmi: {:.4f}'.format(
                acc, f1_macro, f1_micro), flush=True)
            if args.ovr is not None or args.classes or args.val_classes:
                print(confusion_matrix(y_true, y_pred))
            print()

            if args.eval:
                break

            if acc > best_acc:
                best_acc = acc
                best_ep = ep
                if args.out:
                    if not args.not_save:
                        model_f = config.model_path / folder / args.out
                        model_f.parent.mkdir(parents=True, exist_ok=True)
                        model_f = model_f.with_suffix('.pt')
                        torch.save({
                            'model': model.state_dict(),
                        }, model_f)
                    ana_f = config.ana_path / folder / args.out
                    ana_f.parent.mkdir(parents=True, exist_ok=True)
                    ana_f = ana_f.with_suffix('.pt')
                    torch.save({
                        'y_pred': y_pred,
                        'y_true': y_true,
                    }, ana_f)

            if ep - best_ep > args.patience:
                break

    if args.full_epoch > 0:
        model_f = config.model_path / folder / args.out
        model_f.parent.mkdir(parents=True, exist_ok=True)
        model_f = model_f.with_suffix('.pt')
        torch.save({
            'model': model.state_dict(),
        }, model_f)
        print('Save model trained on full dataset')
    else:
        print('Best epoch: {}'.format(best_ep))
        print('Best accuracy: {}'.format(best_acc))
    print('Training time: {}'.format(perf_counter() - s_time))
