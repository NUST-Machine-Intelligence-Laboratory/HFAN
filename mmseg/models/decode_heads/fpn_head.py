import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      build_activation_layer, kaiming_init)
from mmcv.runner import Sequential
from ..utils import SelfAttentionBlock as _SelfAttentionBlock


@HEADS.register_module()
class FPNHead(BaseDecodeHead):
    def __init__(self, feature_strides, **kwargs):
        super(FPNHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.ofam1 = OFAM(channels=self.in_channels[0], r=self.in_channels[0] // 16)
        self.ofam2 = OFAM(channels=self.in_channels[1], r=self.in_channels[1] // 16)
        self.ofam3 = OFAM(channels=self.in_channels[2], r=self.in_channels[2] // 16)
        self.ofam4 = OFAM(channels=self.in_channels[3], r=self.in_channels[3] // 16)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        im1, fw1 = split_batches(c1)
        im2, fw2 = split_batches(c2)
        im3, fw3 = split_batches(c3)
        im4, fw4 = split_batches(c4)
        ofa1 = self.ofam1(im1, fw1)
        ofa2 = self.ofam2(im2, fw2)
        ofa3 = self.ofam3(im3, fw3)
        ofa4 = self.ofam4(im4, fw4)
        x = (ofa1, ofa2, ofa3, ofa4)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.cls_seg(output)
        return output


# object-contextual feature alignment
class OFAM(nn.Module):
    def __init__(self,
                 channels=64,
                 r=4,
                 conv_cfg=None,
                 # norm_cfg=dict(type='SyncBN', requires_grad=True),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')
                 ):
        super(OFAM, self).__init__()
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

        self.object_context_block = ObjectAttentionBlock(
            channels=channels,
            inter_channels=inter_channels,
            scale=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            # norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=act_cfg)
        self.spatial_gather_module = SpatialGatherModule(scale=1)

        self.object_pred = ConvModule(channels, 2, kernel_size=1)

    def forward(self, im, fw):
        im_pred = self.object_pred(im)
        # fw_pred = self.object_pred(fw)
        im_context = self.spatial_gather_module(im, im_pred)
        # fw_context = self.spatial_gather_module(im, fw_pred)
        im_object_context = self.object_context_block(im, im_context)
        fw_object_context = self.object_context_block(fw, im_context)

        foc = im_object_context + fw_object_context
        foc_ca = self.ca_layers(foc)
        foc_pa = self.pa_layers(foc)
        foc_cp = foc_ca + foc_pa
        w = self.sigmoid(foc_cp)

        xo = 2 * im_object_context * w + 2 * fw_object_context * (1 - w)
        return xo


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
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


class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, channels, inter_channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
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
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


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