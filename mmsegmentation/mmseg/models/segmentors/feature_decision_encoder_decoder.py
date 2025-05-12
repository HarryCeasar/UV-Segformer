# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


@MODELS.register_module()
class FeatureDecisionEncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone1: ConfigType,
                 backbone2: ConfigType,
                 backbone3: ConfigType,
                 decode_head1: ConfigType,
                 decode_head2: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head1: OptConfigType = None,
                 auxiliary_head2: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone1.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone1.pretrained = pretrained
            assert backbone2.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone2.pretrained = pretrained
            assert backbone3.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone3.pretrained = pretrained
        self.backbone1 = MODELS.build(backbone1)    # 真彩色影像特征提取encoder,in_channel=3
        self.backbone2 = MODELS.build(backbone2)    # 光谱指标影像特征提取encoder,in_channel=4
        self.backbone3 = MODELS.build(backbone3)    # 建筑指标影像特征提取encoder,in_channel=4
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head1, decode_head2)
        self._init_auxiliary_head(auxiliary_head1, auxiliary_head2)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def with_decode_head(self) -> bool:
        return (self.decode_head1 is not None) and (self.decode_head2 is not None)

    def _init_decode_head(self, decode_head1: ConfigType, decode_head2: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head1 = MODELS.build(decode_head1)
        self.decode_head2 = MODELS.build(decode_head2)
        self.align_corners = self.decode_head1.align_corners
        self.num_classes = self.decode_head1.num_classes
        self.out_channels = self.decode_head1.out_channels

    def _init_auxiliary_head(self, auxiliary_head1: ConfigType, auxiliary_head2: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head1 is not None:
            if isinstance(auxiliary_head1, list):
                self.auxiliary_head1 = nn.ModuleList()
                for head_cfg in auxiliary_head1:
                    self.auxiliary_head1.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head1 = MODELS.build(auxiliary_head1)
        if auxiliary_head2 is not None:
            if isinstance(auxiliary_head2, list):
                self.auxiliary_head2 = nn.ModuleList()
                for head_cfg in auxiliary_head2:
                    self.auxiliary_head2.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head2 = MODELS.build(auxiliary_head2)

    # # 真彩色影像特征提取encoder
    # def extract_feat1(self, inputs: Tensor) -> List[Tensor]:
    #     """Extract features from images."""
    #     x = self.backbone1(inputs)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x
    
    # # 光谱指标影像特征提取encoder
    # def extract_feat2(self, inputs: Tensor) -> List[Tensor]:
    #     """Extract features from images."""
    #     x = self.backbone2(inputs)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x
    
    # # 建筑指标影像特征提取encoder
    # def extract_feat3(self, inputs: Tensor) -> List[Tensor]:
    #     """Extract features from images."""
    #     x = self.backbone3(inputs)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x

    def extract_feat(self, inputs1: Tensor, inputs2: Tensor, inputs3: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x1 = self.backbone3(inputs1)
        x2 = self.backbone3(inputs2)
        x3 = self.backbone3(inputs3)
        if self.with_neck:
            x1 = self.neck(x1)
            x2 = self.neck(x2)
            x3 = self.neck(x3)
        return x1, x2, x3

    # def encode_decode1(self, inputs: Tensor,
    #                   batch_img_metas: List[dict]) -> Tensor:
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     # 输入图像是7波段，前3波段为真彩色，后面4波段为光谱指标
    #     # 将输入图像进行切分
    #     inputs1, inputs2 = torch.split(inputs, [self.backbone1.in_channels, self.backbone2.in_channels], dim=1)
    #     # 真彩色影像特征提取
    #     outs1 = self.extract_feat1(inputs1)
    #     # 光谱指标影像特征提取
    #     outs2 = self.extract_feat2(inputs2)
    #     # 特征融合
    #     outs = [torch.cat((out1, out2), dim=1) for out1, out2 in zip(outs1, outs2)]
    #     # 分割概率计算
    #     seg_logits = self.decode_head1.predict(outs, batch_img_metas,
    #                                           self.test_cfg)
    #     return seg_logits

    # def encode_decode2(self, inputs: Tensor,
    #                   batch_img_metas: List[dict]) -> Tensor:
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     # 输入图像是7波段，前3波段为真彩色，后面4波段为建筑指标
    #     # 将输入图像进行切分
    #     inputs1, inputs3 = torch.split(inputs, [self.backbone1.in_channels, self.backbone3.in_channels], dim=1)
    #     # 真彩色影像特征提取
    #     outs1 = self.extract_feat1(inputs1)
    #     # 建筑指标影像特征提取
    #     outs3 = self.extract_feat3(inputs3)
    #     # 特征融合
    #     outs = [torch.cat((out1, out3), dim=1) for out1, out3 in zip(outs1, outs3)]
    #     # 分割概率计算
    #     seg_logits = self.decode_head2.predict(outs, batch_img_metas,
    #                                           self.test_cfg)
    #     return seg_logits

    def encode_decode(self, spectra_inputs: Tensor, building_inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # 输入图像是7波段，前3波段为真彩色，后面4波段为建筑指标
        # 将输入图像进行切分
        inputs1, inputs2 = torch.split(spectra_inputs, [self.backbone1.in_channels, self.backbone2.in_channels], dim=1)
        inputs1, inputs3 = torch.split(building_inputs, [self.backbone1.in_channels, self.backbone3.in_channels], dim=1)
        # 真彩色影像特征提取
        outs1, outs2, outs3 = self.extract_feat(inputs1, inputs2, inputs3)
        # 特征融合
        spectra_outs = [torch.cat((out1, out2), dim=1) for out1, out2 in zip(outs1, outs2)]
        building_outs = [torch.cat((out1, out3), dim=1) for out1, out3 in zip(outs1, outs3)]
        # 分割概率计算
        seg_logits1 = self.decode_head2.predict(spectra_outs, batch_img_metas,
                                              self.test_cfg)
        seg_logits2 = self.decode_head2.predict(building_outs, batch_img_metas,
                                              self.test_cfg)
        return seg_logits1, seg_logits2



    def _decode_head_forward_train(self, inputs1: List[Tensor], inputs2: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses1 = dict()
        loss_decode1 = self.decode_head1.loss(inputs1, data_samples,
                                            self.train_cfg)
        losses1.update(add_prefix(loss_decode1, 'decode'))

        losses2 = dict()
        loss_decode2 = self.decode_head2.loss(inputs2, data_samples,
                                            self.train_cfg)
        losses2.update(add_prefix(loss_decode2, 'decode'))

        return losses1, losses2
    
    # def _decode_head_forward_train2(self, inputs: List[Tensor],
    #                                data_samples: SampleList) -> dict:
    #     """Run forward function and calculate loss for decode head in
    #     training."""
    #     losses = dict()
    #     loss_decode = self.decode_head2.loss(inputs, data_samples,
    #                                         self.train_cfg)

    #     losses.update(add_prefix(loss_decode, 'decode'))
    #     return losses

    def _auxiliary_head_forward_train(self, inputs1: List[Tensor], inputs2: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses1 = dict()
        if isinstance(self.auxiliary_head1, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head1):
                loss_aux1 = aux_head.loss(inputs1, data_samples, self.train_cfg)
                losses1.update(add_prefix(loss_aux1, f'aux_{idx}'))
        else:
            loss_aux1 = self.auxiliary_head1.loss(inputs1, data_samples,
                                                self.train_cfg)
            losses1.update(add_prefix(loss_aux1, 'aux'))

        losses2 = dict()
        if isinstance(self.auxiliary_head2, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head2):
                loss_aux2 = aux_head.loss(inputs2, data_samples, self.train_cfg)
                losses2.update(add_prefix(loss_aux2, f'aux_{idx}'))
        else:
            loss_aux2 = self.auxiliary_head2.loss(inputs2, data_samples,
                                                self.train_cfg)
            losses2.update(add_prefix(loss_aux2, 'aux'))

        return losses1, losses2
    
    # def _auxiliary_head_forward_train2(self, inputs: List[Tensor],
    #                                   data_samples: SampleList) -> dict:
    #     """Run forward function and calculate loss for auxiliary head in
    #     training."""
    #     losses = dict()
    #     if isinstance(self.auxiliary_head2, nn.ModuleList):
    #         for idx, aux_head in enumerate(self.auxiliary_head2):
    #             loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
    #             losses.update(add_prefix(loss_aux, f'aux_{idx}'))
    #     else:
    #         loss_aux = self.auxiliary_head2.loss(inputs, data_samples,
    #                                             self.train_cfg)
    #         losses.update(add_prefix(loss_aux, 'aux'))

    #     return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 输入图像是11波段，前3波段为真彩色，中间4波段为光谱指标，后面4波段为建筑指标
        # 将输入图像进行切分
        inputs1, inputs2, inputs3 = torch.split(inputs, [self.backbone1.in_channels, self.backbone2.in_channels, self.backbone3.in_channels], dim=1)
        # 真彩色影像特征提取
        x1, x2, x3 = self.extract_feat(inputs1, inputs2, inputs3)
        # 光谱分支特征融合
        spectra_outs = [torch.cat((out1, out2), dim=1) for out1, out2 in zip(x1, x2)]
        # 建筑分支特征融合
        building_outs = [torch.cat((out1, out3), dim=1) for out1, out3 in zip(x1, x3)]

        losses = dict()

        # 光谱分支loss
        loss_decode1, loss_decode2 = self._decode_head_forward_train(spectra_outs, building_outs, data_samples)
        # 加权平均
        loss_decode = dict()
        for key in loss_decode1.keys():
            loss_decode[key] = loss_decode1[key] * 0.5 + loss_decode2[key] * 0.5
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            # 光谱分支loss
            loss_aux1, loss_aux2 = self._auxiliary_head_forward_train(spectra_outs, building_outs, data_samples)
            # 加权平均
            loss_aux = dict()
            for key in loss_aux1.keys():
                loss_aux[key] = loss_aux1[key] * 0.5 + loss_aux2[key] * 0.5
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        # 输入图像是11波段，前3波段为真彩色，中间4波段为光谱指标，后面4波段为建筑指标
        # 将输入图像进行切分
        inputs1, inputs2, inputs3 = torch.split(inputs, [self.backbone1.in_channels, self.backbone2.in_channels, self.backbone3.in_channels], dim=1)
        # 真彩色与光谱指标融合为光谱分支输入
        spectra_inputs = torch.cat((inputs1, inputs2), dim=1)
        # 真彩色与建筑指标融合为建筑分支输入
        building_inputs = torch.cat((inputs1, inputs3), dim=1)
        
        if data_samples is not None:
            batch_img_metas1 = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas1 = [
                dict(
                    ori_shape=spectra_inputs.shape[2:],
                    img_shape=spectra_inputs.shape[2:],
                    pad_shape=spectra_inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * spectra_inputs.shape[0]
        
        if data_samples is not None:
            batch_img_metas2 = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas2 = [
                dict(
                    ori_shape=building_inputs.shape[2:],
                    img_shape=building_inputs.shape[2:],
                    pad_shape=building_inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * building_inputs.shape[0]

        # 光谱分支分割概率
        seg_logits1, seg_logits2 = self.inference(spectra_inputs, building_inputs, batch_img_metas1)
        # 加权平均
        weight = torch.full(seg_logits1.shape, 0.5)
        seg_logits = seg_logits1 * weight + seg_logits2 * weight

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        # 输入图像是11波段，前3波段为真彩色，中间4波段为光谱指标，后面4波段为建筑指标
        # 将输入图像进行切分
        inputs1, inputs2, inputs3 = torch.split(inputs, [self.backbone1.in_channels, self.backbone2.in_channels, self.backbone3.in_channels], dim=1)
        # 真彩色影像特征提取
        x1, x2, x3 = self.extract_feat(inputs1, inputs2, inputs3)
        # 光谱分支特征融合
        spectra_outs = [torch.cat((out1, out2), dim=1) for out1, out2 in zip(x1, x2)]
        # 建筑分支特征融合
        building_outs = [torch.cat((out1, out3), dim=1) for out1, out3 in zip(x1, x3)]

        # 两个分支的向前传播
        return self.decode_head1.forward(spectra_outs), self.decode_head2.forward(building_outs)

    def slide_inference(self, inputs1: Tensor, inputs2: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs1.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds1 = inputs1.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat1 = inputs1.new_zeros((batch_size, 1, h_img, w_img))
        preds2 = inputs2.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat2 = inputs2.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img1 = inputs1[:, :, y1:y2, x1:x2]
                crop_img2 = inputs2[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit1, crop_seg_logit2 = self.encode_decode(crop_img1, crop_img2, batch_img_metas)
                preds1 += F.pad(crop_seg_logit1,
                               (int(x1), int(preds1.shape[3] - x2), int(y1),
                                int(preds1.shape[2] - y2)))
                preds2 += F.pad(crop_seg_logit2,
                               (int(x1), int(preds2.shape[3] - x2), int(y1),
                                int(preds2.shape[2] - y2)))

                count_mat1[:, :, y1:y2, x1:x2] += 1
                count_mat2[:, :, y1:y2, x1:x2] += 1
        assert (count_mat1 == 0).sum() == 0
        seg_logits1 = preds1 / count_mat1
        seg_logits2 = preds2 / count_mat2

        return seg_logits1, seg_logits2
    
    # def slide_inference2(self, inputs: Tensor,
    #                     batch_img_metas: List[dict]) -> Tensor:
    #     """Inference by sliding-window with overlap.

    #     If h_crop > h_img or w_crop > w_img, the small patch will be used to
    #     decode without padding.

    #     Args:
    #         inputs (tensor): the tensor should have a shape NxCxHxW,
    #             which contains all images in the batch.
    #         batch_img_metas (List[dict]): List of image metainfo where each may
    #             also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
    #             'ori_shape', and 'pad_shape'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

    #     Returns:
    #         Tensor: The segmentation results, seg_logits from model of each
    #             input image.
    #     """

    #     h_stride, w_stride = self.test_cfg.stride
    #     h_crop, w_crop = self.test_cfg.crop_size
    #     batch_size, _, h_img, w_img = inputs.size()
    #     out_channels = self.out_channels
    #     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    #     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    #     preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
    #     count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             y1 = h_idx * h_stride
    #             x1 = w_idx * w_stride
    #             y2 = min(y1 + h_crop, h_img)
    #             x2 = min(x1 + w_crop, w_img)
    #             y1 = max(y2 - h_crop, 0)
    #             x1 = max(x2 - w_crop, 0)
    #             crop_img = inputs[:, :, y1:y2, x1:x2]
    #             # change the image shape to patch shape
    #             batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
    #             # the output of encode_decode is seg logits tensor map
    #             # with shape [N, C, H, W]
    #             crop_seg_logit = self.encode_decode2(crop_img, batch_img_metas)
    #             preds += F.pad(crop_seg_logit,
    #                            (int(x1), int(preds.shape[3] - x2), int(y1),
    #                             int(preds.shape[2] - y2)))

    #             count_mat[:, :, y1:y2, x1:x2] += 1
    #     assert (count_mat == 0).sum() == 0
    #     seg_logits = preds / count_mat

    #     return seg_logits

    def whole_inference(self, inputs1: Tensor, inputs2: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits1, seg_logits2 = self.encode_decode(inputs1, inputs2, batch_img_metas)

        return seg_logits1, seg_logits2
    
    # def whole_inference2(self, inputs: Tensor,
    #                     batch_img_metas: List[dict]) -> Tensor:
    #     """Inference with full image.

    #     Args:
    #         inputs (Tensor): The tensor should have a shape NxCxHxW, which
    #             contains all images in the batch.
    #         batch_img_metas (List[dict]): List of image metainfo where each may
    #             also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
    #             'ori_shape', and 'pad_shape'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

    #     Returns:
    #         Tensor: The segmentation results, seg_logits from model of each
    #             input image.
    #     """

    #     seg_logits = self.encode_decode2(inputs, batch_img_metas)

    #     return seg_logits

    def inference(self, inputs1: Tensor, inputs2: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit1, seg_logit2 = self.slide_inference(inputs1, inputs2, batch_img_metas)
        else:
            seg_logit1, seg_logit2 = self.whole_inference(inputs1, inputs2, batch_img_metas)

        return seg_logit1, seg_logit2
    
    # def inference2(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
    #     """Inference with slide/whole style.

    #     Args:
    #         inputs (Tensor): The input image of shape (N, 3, H, W).
    #         batch_img_metas (List[dict]): List of image metainfo where each may
    #             also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
    #             'ori_shape', 'pad_shape', and 'padding_size'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

    #     Returns:
    #         Tensor: The segmentation results, seg_logits from model of each
    #             input image.
    #     """
    #     assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
    #         f'Only "slide" or "whole" test mode are supported, but got ' \
    #         f'{self.test_cfg["mode"]}.'
    #     ori_shape = batch_img_metas[0]['ori_shape']
    #     if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
    #         print_log(
    #             'Image shapes are different in the batch.',
    #             logger='current',
    #             level=logging.WARN)
    #     if self.test_cfg.mode == 'slide':
    #         seg_logit = self.slide_inference2(inputs, batch_img_metas)
    #     else:
    #         seg_logit = self.whole_inference2(inputs, batch_img_metas)

    #     return seg_logit


    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        
        # 输入图像是11波段，前3波段为真彩色，中间4波段为光谱指标，后面4波段为建筑指标
        # 将输入图像进行切分
        inputs1, inputs2, inputs3 = torch.split(inputs, [self.backbone1.in_channels, self.backbone2.in_channels, self.backbone3.in_channels], dim=1)
        # 真彩色与光谱指标融合为光谱分支输入
        spectra_inputs = torch.cat((inputs1, inputs2), dim=1)
        # 真彩色与建筑指标融合为建筑分支输入
        building_inputs = torch.cat((inputs1, inputs3), dim=1)

        # 光谱分支分割概率
        seg_logit1, seg_logit2 = self.inference(spectra_inputs[0], building_inputs[0], batch_img_metas[0], rescale)
        # 加权平均
        weight = torch.full(seg_logit1.shape, 0.5)
        seg_logit = seg_logit1 * weight + seg_logit2 * weight

        for i in range(1, len(inputs)):
            cur_seg_logit1, cur_seg_logit2 = self.inference(spectra_inputs[i], building_inputs[i], batch_img_metas[i],
                                           rescale)
            cur_seg_logit = cur_seg_logit1 * weight + cur_seg_logit2 * weight
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
