# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
# import matplotlib.pyplot as plt
# import numpy as np
from .csp_v7 import CSP_v77


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        #return_layers = {"stem": "0", "dark2": "1", "dark3": "2", "dark4": "3"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):

        # print(len(x))
        # for i in range(len(x)):
        #     print(x[i].shape)
        # #     img = torchvision.utils.make_grid(x[i]).cpu().numpy()
        # #     plt.imshow(np.transpose(x[i].cpu(),(1,2,0)))
        # #     plt.show()

        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, 
                 name: str,
                 pretrained: bool,
                 dilation: bool,
                 norm_type: str):
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=norm_layer)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        super().__init__(backbone, num_channels)

class Backbone_csp_v7(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self):

        backbone = CSP_v77(transition_channels=32, block_channels=32, n=4, phi='l', pretrained=True)
        num_channels = 1024
        super().__init__(backbone, num_channels)


def build_resnet(model_name='resnet18', pretrained=False, norm_type='BN', res5_dilation=False):
    backbone = Backbone(model_name, 
                        pretrained, 
                        dilation=res5_dilation,
                        norm_type=norm_type)
    print(type(backbone.num_channels))
    print(type(backbone))

    return backbone, backbone.num_channels



def build_CSP_v77():
    backbone = Backbone_csp_v7()


    return backbone, backbone.num_channels


if __name__ == '__main__':
    #model, feat_dim = build_resnet(model_name='resnet50', pretrained=False, res5_dilation=True)
    model, feat_dim = build_CSP_v77()

    x = torch.randn(2, 3, 320, 320)
    output = model(x)
    for y in output:
        print(y.size())
