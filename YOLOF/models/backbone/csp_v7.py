import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

from torch.nn.modules.batchnorm import _BatchNorm



def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Block, self).__init__()
        c_ = int(c2 * e)
        
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        
        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
            
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)
    
class Transition(nn.Module):
    def __init__(self, c1, c2):
        super(Transition, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)
        
        self.mp  = MP()

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)
        
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)

        return torch.cat([x_2, x_1], 1)
class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class CSP_v77(nn.Module):
    def __init__(self, transition_channels, block_channels, n, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#
        ids = {
            'l' : [-1, -3, -5, -6],
            'x' : [-1, -3, -5, -7, -8], 
        }[phi]

        self.stem = nn.Sequential(
            Conv(3, transition_channels, 3, 1),
            Conv(transition_channels, transition_channels * 2, 3, 2),
            Conv(transition_channels * 2, transition_channels * 2, 3, 1),
        )
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids),
        )
        self.dark3 = nn.Sequential(
            Transition(transition_channels * 8, transition_channels * 4),
            Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids),
        )
        self.dark4 = nn.Sequential(
            Transition(transition_channels * 16, transition_channels * 8),
            Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )
        self.dark5 = nn.Sequential(
            Transition(transition_channels * 32, transition_channels * 16),
            Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )
        # self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 32)
        if pretrained:
            url = {
                "l" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./checkpoints/")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):

        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        # x = self.dark5(x)
        # out = self.sppcspc(x)
        out = x
        x_dim = 1024

        return out, x_dim
'''
def build_CSP_v77():
    backbone = Backbone_csp_v7()
    print(type(backbone.num_channels))
    print(type(backbone))

    return backbone, backbone.num_channels

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        #return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        return_layers = {"dark2": "0", "dark3": "1", "dark4": "2", "dark5": "3"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        xs = self.body(x)
        fmp_list = []
        for name, fmp in xs.items():
            fmp_list.append(fmp)

        return fmp_list

class Backbone_csp_v7(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self):

        backbone = CSP_v77(transition_channels=32, block_channels=32, n=4, phi='l', pretrained=False)
        num_channels = 1024
        super().__init__(backbone, num_channels)
'''

def main():
    backbone = CSP_v77(transition_channels=32,block_channels=32,n=4,phi='l',pretrained=False)
    a = torch.rand(8,3,640,640)
    b = torch.rand(3,640,640).unsqueeze(0)
    print(backbone.forward(a)[0].shape)

if __name__ == '__main__':
    main()