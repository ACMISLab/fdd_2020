# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
#from .detectors_resnest import DetectoRS_ResNeSt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2, PyramidVisionTransformerV2PSA_cbam1, PyramidVisionTransformerV2PSA_cbam2, PyramidVisionTransformerV2PSA_cbam0, PyramidVisionTransformerV2CF_0, PyramidVisionTransformerV2CF_1, PyramidVisionTransformerV2CF_2
from .regnet import RegNet
from .res2net import Res2Net
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
# from .gvt import pcpvt_small,alt_gvt_small,pcpvt_base,alt_gvt_base,pcpvt_large,alt_gvt_large

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer', 'PyramidVisionTransformerV2PSA_cbam1', 'PyramidVisionTransformerV2PSA_cbam2', 'PyramidVisionTransformerV2PSA_cbam0',
    'PyramidVisionTransformerV2CF_0', 'PyramidVisionTransformerV2CF_1', 'PyramidVisionTransformerV2CF_2', 'EfficientNet'
]
