# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn

from util.misc_sptsv2 import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone

from .encoder_decoder import build_transformer
import pdb

class SPTSv2(nn.Module):
    def __init__(self, backbone, transformer, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes

        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.vocab_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.num_classes = num_classes

    def forward(self, samples: NestedTensor, sequence, sequence_reg, text_length = 25):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - sequence: the input sequence for locating in the first decoder layer.
               - sequence_reg: the input sequence for recognizing in the second decoder layer.

            It returns a dict with the following elements:
               - out: prediction of location and recognition 
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        
        src, mask = features[-1].decompose()
        
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, None, pos[-1], sequence,sequence_reg, self.vocab_embed)
        if hs == None:
            return None
        if self.training:
           assert hs[2].size(2) == (text_length+2)
           out_point = self.vocab_embed(hs[0])[-1]
           out_label = self.vocab_embed(hs[2])[-1][:,1:text_length+1].reshape(src.size(0),-1,self.num_classes)
           return out_point,out_label
        else:
            out_point = hs[0].reshape(src.size(0),-1,2)
            out_label = hs[2].reshape(src.size(0),-1,text_length)
            out = torch.cat([out_point,out_label],dim=-1).view(src.size(0),-1)
            value = hs[3].reshape(src.size(0),-1,2)
            value_reg = hs[1].reshape(src.size(0),-1,text_length)
            out_v = torch.cat([value,value_reg],dim=-1).view(src.size(0),-1)
            rec_score = hs[4].reshape(src.size(0),-1,hs[4].shape[-1])
            return out, out_v, rec_score

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    num_classes = args.padding_index + 1
    model = SPTSv2(backbone, transformer, num_classes)

    weight = torch.ones(num_classes)
    weight[args.end_index] = 0.01; weight[args.noise_index] = 0.01
    criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=args.padding_index)
    criterion.to(device)

    return model, criterion
