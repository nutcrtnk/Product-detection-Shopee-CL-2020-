
from torchvision import transforms as T
import config
from PIL import Image
from pathlib import Path
import torch.utils.data as D
import torch
import re


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def mapping(dict):
    def map_func(x):
        return dict[x]
    return map_func


class SPDataset(D.Dataset):

    regex = re.compile('[^\da-zA-Z]')
    regex2 = re.compile('\d')

    @staticmethod
    def process(x):
        x = x.lower()
        x = SPDataset.regex.sub('', x)
        x = SPDataset.regex2.sub('#', x)
        return x

    def __init__(self, path, filenames=None, img_size=256, classes=None, options='', test=False, filter_size=0, constrastive=False, ymap=int, use_words=None):
        super().__init__()
        self.path = path
        self.filenames = filenames
        self.img_size = img_size
        self.labels = set()
        if self.filenames is None:
            self.filenames = []
            p = config.data_path / self.path
            for pic_f in p.glob(r'*/*'):
                if pic_f.suffix.lower() in ['.jpg', '.png']:
                    self.filenames.append('{}/{}'.format(pic_f.parent.name, pic_f.name))
            print(len(self.filenames))
        for fn in self.filenames:
            self.labels.add(fn.split('/')[0])
        self.labels = list(self.labels)

        if classes is not None:
            self.labels = classes
            self.filenames = [f for f in self.filenames if int(f.split('/')[0]) in classes]
            ydict = {c: i for i, c in enumerate(classes)}
            self.ymap = mapping(ydict)
        elif ymap == 'auto':
            ydict = {c: i for i, c in enumerate(self.labels)}
            self.ymap = mapping(ydict)
        else:
            self.ymap = ymap

        if filter_size > 0:
            self.filter_by_size(filter_size)

        transform = [T.Resize(self.img_size)]
        
        if constrastive:
            transform.append(T.RandomResizedCrop(int(self.img_size * 0.9), scale=(0.2, 1.0)))
        elif 'r' in options:
            transform.append(T.RandomCrop(int(self.img_size * 0.9)))
        else:
            transform.append(T.CenterCrop(int(self.img_size * 0.9)))
            # transform.append(T.CenterCrop(self.img_size))

        if 'f' in options:
            transform.append(T.RandomHorizontalFlip())
        if 'c' in options:
            transform.append(T.RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8))
        if 'g' in options:
            transform.append(T.RandomGrayscale(0.2))
        transform += [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if 'e' in options:
            transform.append(T.RandomErasing())
        self.transform = T.Compose(transform)
        if constrastive:
            self.transform = TwoCropTransform(self.transform)
        self.test = test
        if use_words is not None:
            with open((config.data_text_path / use_words).with_suffix('.txt')) as f:
                self.words = {w: i for i, w in enumerate([x.strip() for x in f.read().splitlines()])}
        else:
            self.words = None

    def filter_by_size(self, img_size):
        new_filenames = []
        for index in range(len(self.filenames)):
            fn = config.data_path / self.path / self.filenames[index]
            img = Image.open(fn)
            width, height = img.size
            if width >= img_size or height >= img_size:
                new_filenames.append(self.filenames[index])
        self.filenames = new_filenames
    
    def __getitem__(self, index):
        fn = config.data_path / self.path / self.filenames[index]
        img = Image.open(fn).convert('RGB')
        X = self.transform(img)
        if self.words is not None:
            x_words = torch.zeros(len(self.words))
            fn_text = config.data_text_path / ('test' if self.test else 'train') / self.filenames[index]
            fn_text = fn_text.with_suffix('.txt')
            with open(fn_text, encoding="utf-8") as f:
                words = set([self.process(x) for x in f.read().strip().split(' ')])
                for w in words:
                    if w in self.words:
                        x_words[self.words[w]] = 1.
            X = [X, x_words]

        if self.test:
            return X
        y = self.ymap(fn.parent.name)
        return X, y

    def __len__(self):
        return len(self.filenames)