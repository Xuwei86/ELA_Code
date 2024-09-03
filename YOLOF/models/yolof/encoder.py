import torch.nn as nn
from ..basic.conv import Conv
from utils import weight_init
from .attention import se_block, CA_Block, CA, CA2, ELA72, ELA73, ELA10,  sa_layer, SGE, cbam_block,  EC_SA3, EC_SA33, EC_SA39, eac
attention_block = [se_block, CA_Block, CA, CA2, ELA72,  ELA73,  ELA10,  sa_layer, SGE, cbam_block, EC_SA3, EC_SA33, EC_SA39,  eac ]


# Dilated Encoder
class Bottleneck(nn.Module):
    def __init__(self, 
                 in_dim, 
                 dilation=1, 
                 expand_ratio=0.25,
                 act_type='relu',
                 norm_type='BN'):
        super(Bottleneck, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.branch = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation, d=dilation, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        return x + self.branch(x)
        #return self.feat512(x + self.branch(x))

class Bottleneck_attention(nn.Module):
    def __init__(self,
                 in_dim,
                 dilation=1,
                 expand_ratio=0.25,
                 act_type='relu',
                 norm_type='BN'):
        super(Bottleneck_attention, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.branch = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation, d=dilation, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type, norm_type=norm_type)
        )
        self.phi = 0

        self.feat512 = attention_block[self.phi-1](512)

    def forward(self, x):
        if 1 <= self.phi and self.phi <= 19:
            x0 = x
            x = self.feat512(x)
            x = x + x0
            y = x + self.branch(x)
            y1 = self.feat512(y)
            return y + y1
        else:
            return x + self.branch(x)


class DilatedEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 expand_ratio=0.25, 
                 dilation_list=[2, 4, 6, 8],
                 act_type='relu',
                 norm_type='BN'):
        super(DilatedEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None, norm_type=norm_type),
            Conv(out_dim, out_dim, k=3, p=1, act_type=None, norm_type=norm_type)
        )
        self.phi = 0
        self.feat2048 = attention_block[self.phi-1](2048)
        self.feat512 = attention_block[self.phi-1](512)

        encoders = []
        # for d in dilation_list:
        #     encoders.append(Bottleneck(in_dim=out_dim,  dilation=d, expand_ratio=expand_ratio,
        #                                act_type=act_type,  norm_type=norm_type))
        # self.encoders = nn.Sequential(*encoders)
        for d in dilation_list:
            if d != dilation_list[-1]:
                encoders.append(Bottleneck_attention(in_dim=out_dim,  dilation=d, expand_ratio=expand_ratio,
                                       act_type=act_type,  norm_type=norm_type))
            else:
                encoders.append(Bottleneck(in_dim=out_dim, dilation=d, expand_ratio=expand_ratio,
                                           act_type=act_type, norm_type=norm_type))
        self.encoders = nn.Sequential(*encoders)


        self._init_weight()

    def _init_weight(self):
        for m in self.projector:
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
                weight_init.c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if 1 <= self.phi and self.phi <= 19:
            y0 = x
            x = self.feat2048(x)   #add CA attention
            x = x + y0
        x = self.projector(x)
        if 1 <= self.phi and self.phi <= 19:
            y1 = x
            x = self.feat512(x)  #add CA attention
            x = x + y1
        x = self.encoders(x)

        return x
'''
class DilatedEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.25,
                 dilation_list=[2, 4, 6, 8],
                 act_type='relu',
                 norm_type='BN'):
        super(DilatedEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None, norm_type=norm_type),
            Conv(out_dim, out_dim, k=3, p=1, act_type=None, norm_type=norm_type)
        )


        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(in_dim=out_dim,  dilation=d, expand_ratio=expand_ratio,
                                       act_type=act_type,  norm_type=norm_type))
        self.encoders = nn.Sequential(*encoders)
        for d in dilation_list:
            if d != dilation_list[-1]:
                encoders.append(Bottleneck_attention(in_dim=out_dim,  dilation=d, expand_ratio=expand_ratio,
                                       act_type=act_type,  norm_type=norm_type))
            else:
                encoders.append(Bottleneck(in_dim=out_dim, dilation=d, expand_ratio=expand_ratio,
                                           act_type=act_type, norm_type=norm_type))
        self.encoders = nn.Sequential(*encoders)


        self._init_weight()

    def _init_weight(self):
        for m in self.projector:
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
                weight_init.c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x = self.feat2048(x)   #add CA attention
        x = self.projector(x)
        #x = self.feat512(x)  #add CA attention
        x = self.encoders(x)

        return x
'''

# build encoder
def build_encoder(cfg, in_dim, out_dim):
    print('==============================')
    print('Neck: {}'.format('Dilated Encoder'))
    
    neck = DilatedEncoder(
        in_dim=in_dim,
        out_dim=out_dim,
        expand_ratio=cfg['expand_ratio'],
        dilation_list=cfg['dilation_list'],
        act_type=cfg['encoder_act_type'],
        norm_type=cfg['encoder_norm_type']
        )

    return neck
