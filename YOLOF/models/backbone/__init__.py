from .resnet import build_resnet , build_CSP_v77
from  .csp_v7 import CSP_v77
from  .efficientrep import EfficientRep , build_efficientrep

def build_backbone(cfg, pretrained=True):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if cfg['backbone'] in ['resnet18', 'resnet50', 'resnet101']:
        model, feat_dim = build_resnet(
            model_name=cfg['backbone'], 
            pretrained=pretrained,
            norm_type=cfg['bk_norm_type'],
            res5_dilation=cfg['res5_dilation']
            )

    elif cfg['backbone'] in ['CSP_v77']:
        #model, feat_dim = CSP_v77(transition_channels=32,  block_channels=32, n=4, phi='l', pretrained=False)
        model, feat_dim = build_CSP_v77()
    else:
        model, feat_dim = build_efficientrep()


    return model, feat_dim
