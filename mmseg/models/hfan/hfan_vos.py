import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F

from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      build_activation_layer, kaiming_init)
from mmcv.runner import Sequential

from mmseg.utils import split_images
from mmseg.ops import resize
from .. import builder
from ..builder import BACKBONES
from ..builder import HEADS
from ..decode_heads.decode_head import BaseDecodeHead


@BACKBONES.register_module()
class HFANVOS(nn.Module):
    def __init__(self, backbone_method='mit', **kargs):
        assert isinstance(backbone_method, str), \
            f'merge_method should a str object, but got {type(backbone_method)}'
        self.backbone_method = backbone_method
        super().__init__()
        kargs.update(type=kargs.pop('ori_type'))
        self.backbone = builder.build_backbone(kargs)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)

    def forward(self, x):
        x1, x2 = split_images(x)
        x = merge_batches(x1, x2)
        x = self.backbone(x)

        return x


@HEADS.register_module()
class HFANVOS_Head(BaseDecodeHead):
    # im: only image; fw: only optical flow; ff: image and optical flow
    # fa: hifi baseline of adaptation; fs: hifi baseline of alignment; ofa: hifi
    # select_method = ['im', 'fw', 'ff', 'fa', 'fs', 'ofa']
    def __init__(self, feature_strides, select_method, **kwargs):
        super(HFANVOS_Head, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.select_method = select_method

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        if self.select_method == 'fam':
            # baseline + FAM
            self.fam1 = FAM(channels=c1_in_channels, r=c1_in_channels // 16)
            self.fam2 = FAM(channels=c2_in_channels, r=c2_in_channels // 16)
            self.fam3 = FAM(channels=c3_in_channels, r=c3_in_channels // 16)
            self.fam4 = FAM(channels=c4_in_channels, r=c4_in_channels // 16)
        elif self.select_method == 'fat':
            # baseline + FAT
            self.fat1 = FAT(channels=c1_in_channels, r=c1_in_channels // 16)
            self.fat2 = FAT(channels=c2_in_channels, r=c2_in_channels // 16)
            self.fat3 = FAT(channels=c3_in_channels, r=c3_in_channels // 16)
            self.fat4 = FAT(channels=c4_in_channels, r=c4_in_channels // 16)
        elif self.select_method == 'hfan':
            # hifi: object-contextual feature alignment
            self.hfan1 = HFAN(channels=c1_in_channels, r=c1_in_channels // 16)
            self.hfan2 = HFAN(channels=c2_in_channels, r=c2_in_channels // 16)
            self.hfan3 = HFAN(channels=c3_in_channels, r=c3_in_channels // 16)
            self.hfan4 = HFAN(channels=c4_in_channels, r=c4_in_channels // 16)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        im1, fw1 = split_batches(c1)
        im2, fw2 = split_batches(c2)
        im3, fw3 = split_batches(c3)
        im4, fw4 = split_batches(c4)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = im4.shape
        '''im: only image; fw: only optical flow; 
        base: baseline, fam: baseline + FAM; fat: baseline + FAT; 
        hfan: HFAN'''
        '''methods = ['im', 'fw', 'base', 'fam', 'fat', 'hfan']'''
        if self.select_method == 'im':
            # Image: 4
            _a4 = self.linear_c4(im4).permute(0, 2, 1).reshape(n, -1, im4.shape[2], im4.shape[3])
            _a4 = resize(_a4, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # Image: 3
            _a3 = self.linear_c3(im3).permute(0, 2, 1).reshape(n, -1, im3.shape[2], im3.shape[3])
            _a3 = resize(_a3, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # Image: 2
            _a2 = self.linear_c2(im2).permute(0, 2, 1).reshape(n, -1, im2.shape[2], im2.shape[3])
            _a2 = resize(_a2, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # Image: 1
            _a1 = self.linear_c1(im1).permute(0, 2, 1).reshape(n, -1, im1.shape[2], im1.shape[3])
            # prediction
            _a = self.linear_fuse(torch.cat([_a4, _a3, _a2, _a1], dim=1))
            a = self.dropout(_a)
            a = self.linear_pred(a)
            return a
        elif self.select_method == 'fw':
            # Optical Flow: 4
            _f4 = self.linear_c4(fw4).permute(0, 2, 1).reshape(n, -1, fw4.shape[2], fw4.shape[3])
            _f4 = resize(_f4, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # Optical Flow: 3
            _f3 = self.linear_c3(fw3).permute(0, 2, 1).reshape(n, -1, fw3.shape[2], fw3.shape[3])
            _f3 = resize(_f3, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # Optical Flow: 2
            _f2 = self.linear_c2(fw2).permute(0, 2, 1).reshape(n, -1, fw2.shape[2], fw2.shape[3])
            _f2 = resize(_f2, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # Optical Flow: 1
            _f1 = self.linear_c1(fw1).permute(0, 2, 1).reshape(n, -1, fw1.shape[2], fw1.shape[3])
            # prediction
            _f = self.linear_fuse(torch.cat([_f4, _f3, _f2, _f1], dim=1))
            f = self.dropout(_f)
            f = self.linear_pred(f)
            return f
        elif self.select_method == 'base':
            ff1 = im1 + fw1
            ff2 = im2 + fw2
            ff3 = im3 + fw3
            ff4 = im4 + fw4
            # FF: 4
            _ff4 = self.linear_c4(ff4).permute(0, 2, 1).reshape(n, -1, ff4.shape[2], ff4.shape[3])
            _ff4 = resize(_ff4, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # FF: 3
            _ff3 = self.linear_c3(ff3).permute(0, 2, 1).reshape(n, -1, ff3.shape[2], ff3.shape[3])
            _ff3 = resize(_ff3, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # FF: 2
            _ff2 = self.linear_c2(ff2).permute(0, 2, 1).reshape(n, -1, ff2.shape[2], ff2.shape[3])
            _ff2 = resize(_ff2, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # FF: 1
            _ff1 = self.linear_c1(ff1).permute(0, 2, 1).reshape(n, -1, ff1.shape[2], ff1.shape[3])
            # prediction
            _ff = self.linear_fuse(torch.cat([_ff4, _ff3, _ff2, _ff1], dim=1))
            ff = self.dropout(_ff)
            ff = self.linear_pred(ff)
            return ff
        elif self.select_method == 'fam':
            fa1 = self.fam1(im1, fw1)
            fa2 = self.fam2(im2, fw2)
            fa3 = self.fam3(im3, fw3)
            fa4 = self.fam4(im4, fw4)
            # FAM: 4
            _fa4 = self.linear_c4(fa4).permute(0, 2, 1).reshape(n, -1, fa4.shape[2], fa4.shape[3])
            _fa4 = resize(_fa4, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # FAM: 3
            _fa3 = self.linear_c3(fa3).permute(0, 2, 1).reshape(n, -1, fa3.shape[2], fa3.shape[3])
            _fa3 = resize(_fa3, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # FAM: 2
            _fa2 = self.linear_c2(fa2).permute(0, 2, 1).reshape(n, -1, fa2.shape[2], fa2.shape[3])
            _fa2 = resize(_fa2, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # FAM: 1
            _fa1 = self.linear_c1(fa1).permute(0, 2, 1).reshape(n, -1, fa1.shape[2], fa1.shape[3])
            # prediction
            _fa = self.linear_fuse(torch.cat([_fa4, _fa3, _fa2, _fa1], dim=1))
            fa = self.dropout(_fa)
            fa = self.linear_pred(fa)
            return fa
        elif self.select_method == 'fat':
            fs1 = self.fat1(im1, fw1)
            fs2 = self.fat2(im2, fw2)
            fs3 = self.fat3(im3, fw3)
            fs4 = self.fat4(im4, fw4)
            # FAT: 4
            _fs4 = self.linear_c4(fs4).permute(0, 2, 1).reshape(n, -1, fs4.shape[2], fs4.shape[3])
            _fs4 = resize(_fs4, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # FAT: 3
            _fs3 = self.linear_c3(fs3).permute(0, 2, 1).reshape(n, -1, fs3.shape[2], fs3.shape[3])
            _fs3 = resize(_fs3, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # FAT: 2
            _fs2 = self.linear_c2(fs2).permute(0, 2, 1).reshape(n, -1, fs2.shape[2], fs2.shape[3])
            _fs2 = resize(_fs2, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # FAT: 1
            _fs1 = self.linear_c1(fs1).permute(0, 2, 1).reshape(n, -1, fs1.shape[2], fs1.shape[3])
            # prediction
            _fs = self.linear_fuse(torch.cat([_fs4, _fs3, _fs2, _fs1], dim=1))
            fs = self.dropout(_fs)
            fs = self.linear_pred(fs)
            return fs
        elif self.select_method == 'hfan':
            ofa1 = self.hfan1(im1, fw1)
            ofa2 = self.hfan2(im2, fw2)
            ofa3 = self.hfan3(im3, fw3)
            ofa4 = self.hfan4(im4, fw4)
            # OFA: 4
            _ofa4 = self.linear_c4(ofa4).permute(0, 2, 1).reshape(n, -1, ofa4.shape[2], ofa4.shape[3])
            _ofa4 = resize(_ofa4, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # OFA: 3
            _ofa3 = self.linear_c3(ofa3).permute(0, 2, 1).reshape(n, -1, ofa3.shape[2], ofa3.shape[3])
            _ofa3 = resize(_ofa3, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # OFA: 2
            _ofa2 = self.linear_c2(ofa2).permute(0, 2, 1).reshape(n, -1, ofa2.shape[2], ofa2.shape[3])
            _ofa2 = resize(_ofa2, size=im1.size()[2:], mode='bilinear', align_corners=False)
            # OFA: 1
            _ofa1 = self.linear_c1(ofa1).permute(0, 2, 1).reshape(n, -1, ofa1.shape[2], ofa1.shape[3])
            # prediction
            _ofa = self.linear_fuse(torch.cat([_ofa4, _ofa3, _ofa2, _ofa1], dim=1))
            ofa = self.dropout(_ofa)
            ofa = self.linear_pred(ofa)
            return ofa
        else:
            assert f'select_method is not supported!'


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# (CSS) Category-Specific Semantic
class CSS(nn.Module):
    def __init__(self, scale):
        super(CSS, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        # b, c, h, w = feats.size()
        # print(batch_size, num_classes, height, width)
        # print(b, c, h, w)
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


# (POC) Primary Object Context
class POC(_SelfAttentionBlock):
    def __init__(self, channels, inter_channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(POC, self).__init__(
            key_in_channels=channels,
            query_in_channels=channels,
            channels=inter_channels,
            out_channels=channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            channels * 2,
            channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(POC,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


# baseline + FAM (Feature AlignMent)
class FAM(nn.Module):
    def __init__(self,
                 channels=64,
                 r=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')
                 ):
        super(FAM, self).__init__()
        inter_channels = int(channels // r)

        self.poc = POC(
            channels=channels,
            inter_channels=inter_channels,
            scale=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.css = CSS(scale=1)

        self.object_pred = ConvModule(channels, 2, kernel_size=1)

    def forward(self, im, fw):
        im_pred = self.object_pred(im)
        im_context = self.css(im, im_pred)
        im_object_context = self.poc(im, im_context)
        fw_object_context = self.poc(fw, im_context)

        foc = im_object_context + fw_object_context

        return foc


# baseline + FAT (Feature AdaptaTion)
class FAT(nn.Module):
    def __init__(self,
                 channels=64,
                 r=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')
                 ):
        super(FAT, self).__init__()
        inter_channels = int(channels // r)

        # channel attention
        ca_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        ca_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
        ca_act1 = build_activation_layer(act_cfg)
        ca_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        ca_bn2 = build_norm_layer(norm_cfg, channels)[1]
        ca_layers = [ca_conv1, ca_bn1, ca_act1, ca_conv2, ca_bn2]
        self.ca_layers = Sequential(*ca_layers)
        # pixel attention
        ap = nn.AdaptiveAvgPool2d(1)
        pa_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        pa_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
        pa_act1 = build_activation_layer(act_cfg)
        pa_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        pa_bn2 = build_norm_layer(norm_cfg, channels)[1]
        pa_layers = [ap, pa_conv1, pa_bn1, pa_act1, pa_conv2, pa_bn2]
        self.pa_layers = Sequential(*pa_layers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, im, fw):
        f = im + fw
        f_ca = self.ca_layers(f)
        f_pa = self.pa_layers(f)
        f_cp = f_ca + f_pa
        w = self.sigmoid(f_cp)

        f = 2 * im * w + 2 * fw * (1 - w)
        return f


# HFAN (Hierarchical Feature Alignment Network)
class HFAN(nn.Module):
    def __init__(self,
                 channels=64,
                 r=4,
                 conv_cfg=None,
                 # norm_cfg=dict(type='SyncBN', requires_grad=True),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')
                 ):
        super(HFAN, self).__init__()
        inter_channels = int(channels // r)

        # channel attention
        ca_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        ca_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
        ca_act1 = build_activation_layer(act_cfg)
        ca_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        ca_bn2 = build_norm_layer(norm_cfg, channels)[1]
        ca_layers = [ca_conv1, ca_bn1, ca_act1, ca_conv2, ca_bn2]
        self.ca_layers = Sequential(*ca_layers)
        # pixel attention
        ap = nn.AdaptiveAvgPool2d(1)
        pa_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        pa_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
        pa_act1 = build_activation_layer(act_cfg)
        pa_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        pa_bn2 = build_norm_layer(norm_cfg, channels)[1]
        pa_layers = [ap, pa_conv1, pa_bn1, pa_act1, pa_conv2, pa_bn2]
        self.pa_layers = Sequential(*pa_layers)

        self.sigmoid = nn.Sigmoid()

        self.poc = POC(
            channels=channels,
            inter_channels=inter_channels,
            scale=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            # norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=act_cfg)
        self.css = CSS(scale=1)

        self.object_pred = ConvModule(channels, 2, kernel_size=1)

    def forward(self, im, fw):
        im_pred = self.object_pred(im)
        # fw_pred = self.object_pred(fw)
        im_context = self.css(im, im_pred)
        # fw_context = self.spatial_gather_module(im, fw_pred)
        im_object_context = self.poc(im, im_context)
        fw_object_context = self.poc(fw, im_context)

        foc = im_object_context + fw_object_context
        foc_ca = self.ca_layers(foc)
        foc_pa = self.pa_layers(foc)
        foc_cp = foc_ca + foc_pa
        w = self.sigmoid(foc_cp)

        foc = 2 * im_object_context * w + 2 * fw_object_context * (1 - w)
        return foc


def split_batches(x: Tensor):
    """ Split a 2*B batch of images into two B images per batch,
    in order to adapt to MMSegmentation """

    assert x.ndim == 4, f'expect to have 4 dimensions, but got {x.ndim}'
    batch_size = x.shape[0] // 2
    x1 = x[0:batch_size, ...]
    x2 = x[batch_size:, ...]
    return x1, x2


def merge_batches(x1: Tensor, x2: Tensor):
    """ merge two batches each contains B images into a 2*B batch of images
    in order to adapt to MMSegmentation """

    assert x1.ndim == 4 and x2.ndim == 4, f'expect x1 and x2 to have 4 \
                dimensions, but got x1.dim: {x1.ndim}, x2.dim: {x2.ndim}'
    return torch.cat((x1, x2), dim=0)
