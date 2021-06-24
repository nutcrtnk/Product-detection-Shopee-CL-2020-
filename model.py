import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import geffnet


# efn_net
# https://github.com/rwightman/gen-efficientnet-pytorch

def get_model(model_name, pretrained=True):
    if model_name == 'wrn':
        return models.wide_resnet101_2(pretrained=pretrained, progress=False)
    elif model_name == 'rn':
        return models.resnet152(pretrained=pretrained, progress=False)
    elif model_name == 'rnx':
        return models.resnext101_32x8d(pretrained=pretrained, progress=False)
    elif model_name == 'dn':
        return models.densenet161(pretrained=pretrained, progress=False)
    elif model_name == 'mbn':
        return models.mobilenet_v2(pretrained=pretrained, progress=False)
    elif model_name == 'efn':
        return geffnet.create_model('tf_efficientnet_b5_ns', pretrained=pretrained)
    elif model_name[:4] == 'efn_':
        return geffnet.create_model('tf_efficientnet_{}'.format(model_name[4:]), pretrained=pretrained)
    else:
        raise ValueError


def get_clf_layer(model):
    clf_name, clf_layer = None, None
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            clf_name, clf_layer = n, m
    return clf_name, clf_layer


def set_layer(model, layer_name, new_layer):
    names = layer_name.split('.')
    layer = model
    for name in names[:-1]:
        layer = getattr(layer, name)
    setattr(layer, names[-1], new_layer)

def load_ckpt(model, path):
    model_dict = torch.load(path, map_location='cpu')['model']
    try:
        model.load_state_dict(model_dict)
    except RuntimeError:
        model_dict = {'.'.join(k.split('.')[1:]): v for k, v in model_dict.items()}
        model.load_state_dict(model_dict)


class Iden(nn.Module):
    def forward(self, x):
        return x


class ImgText(nn.Module):

    def __init__(self, encoder, n_words, num_classes):
        super().__init__()
        self.img_encoder = encoder
        self.word_encoder = nn.Linear(n_words, num_classes)

    def forward(self, x):
        y = self.img_encoder(x[0]) + self.word_encoder(x[1])
        return y


class SupCon(nn.Module):
    """backbone + projection head"""
    def __init__(self, encoder, dim_in, feat_dim=128, head='mlp'):
        super(SupCon, self).__init__()
        self.encoder = encoder
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = F.normalize(self.head(self.encoder(x)), dim=1)
        return feat
