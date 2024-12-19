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
class DecisionEncoderDecoder(BaseSegmentor):
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
                 decode_head3: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head1: OptConfigType = None,
                 auxiliary_head2: OptConfigType = None,
                 auxiliary_head3: OptConfigType = None,
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
        if pretrained is not None:
            assert backbone2.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone2.pretrained = pretrained
        if pretrained is not None:
            assert backbone3.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone3.pretrained = pretrained
        self.backbone1 = MODELS.build(backbone1)
        self.backbone2 = MODELS.build(backbone2)
        self.backbone3 = MODELS.build(backbone3)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head1, decode_head2, decode_head3)
        self._init_auxiliary_head(auxiliary_head1, auxiliary_head2, auxiliary_head3)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head
    
    def with_decode_head(self) -> bool:
        return (self.decode_head1 is not None) and (self.decode_head2 is not None) and (self.decode_head3 is not None)
    
    def with_auxiliary_head(self) -> bool:
        return (self.auxiliary_head1 is not None) and (self.auxiliary_head2 is not None) and (self.auxiliary_head3 is not None)

    def _init_decode_head(self, decode_head1: ConfigType, decode_head2: ConfigType, decode_head3: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head1 = MODELS.build(decode_head1)
        self.decode_head2 = MODELS.build(decode_head2)
        self.decode_head3 = MODELS.build(decode_head3)
        self.align_corners = self.decode_head1.align_corners
        self.num_classes = self.decode_head1.num_classes
        self.out_channels = self.decode_head1.out_channels

    def _init_auxiliary_head(self, auxiliary_head1: ConfigType, auxiliary_head2: ConfigType, auxiliary_head3: ConfigType) -> None:
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
        
        if auxiliary_head3 is not None:
            if isinstance(auxiliary_head3, list):
                self.auxiliary_head3 = nn.ModuleList()
                for head_cfg in auxiliary_head3:
                    self.auxiliary_head3.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head3 = MODELS.build(auxiliary_head3)

    def extract_feat(self, img_inputs: Tensor, specrta_inputs: Tensor, building_inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        img_x = self.backbone1(img_inputs)
        spectra_x = self.backbone2(specrta_inputs)
        building_x = self.backbone3(building_inputs)
        if self.with_neck:
            img_x = self.neck(img_x)
            spectra_x = self.neck(spectra_x)
            building_x = self.neck(building_x)
        return img_x, spectra_x, building_x

    def encode_decode(self, img_inputs: Tensor, specrta_inputs: Tensor, building_inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        img_x, spectra_x, building_x = self.extract_feat(img_inputs, specrta_inputs, building_inputs)
        img_seg_logits = self.decode_head1.predict(img_x, batch_img_metas,
                                              self.test_cfg)
        spectra_seg_logits = self.decode_head2.predict(spectra_x, batch_img_metas,
                                              self.test_cfg)
        building_seg_logits = self.decode_head3.predict(building_x, batch_img_metas,
                                              self.test_cfg)

        return img_seg_logits, spectra_seg_logits, building_seg_logits

    def _decode_head_forward_train(self, img_inputs: List[Tensor], spectra_inputs: List[Tensor], building_inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        img_losses = dict()
        loss_decode = self.decode_head1.loss(img_inputs, data_samples,
                                            self.train_cfg)
        
        img_losses.update(add_prefix(loss_decode, 'decode'))

        spectra_losses = dict()
        loss_decode = self.decode_head2.loss(spectra_inputs, data_samples,
                                            self.train_cfg)
        
        spectra_losses.update(add_prefix(loss_decode, 'decode'))

        building_losses = dict()
        loss_decode = self.decode_head3.loss(building_inputs, data_samples,
                                            self.train_cfg)
        
        building_losses.update(add_prefix(loss_decode, 'decode'))

        return img_losses, spectra_losses, building_losses

    def _auxiliary_head_forward_train(self, img_inputs: List[Tensor], spectra_inputs: List[Tensor], building_inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        img_losses = dict()
        if isinstance(self.auxiliary_head1, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head1):
                img_loss_aux = aux_head.loss(img_inputs, data_samples, self.train_cfg)
                img_losses.update(add_prefix(img_loss_aux, f'aux_{idx}'))
        else:
            img_loss_aux = self.auxiliary_head1.loss(img_inputs, data_samples,
                                                self.train_cfg)
            img_losses.update(add_prefix(img_loss_aux, 'aux'))
        
        spectra_losses = dict()
        if isinstance(self.auxiliary_head2, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head2):
                spectra_loss_aux = aux_head.loss(spectra_inputs, data_samples, self.train_cfg)
                spectra_losses.update(add_prefix(spectra_loss_aux, f'aux_{idx}'))
        else:
            spectra_loss_aux = self.auxiliary_head2.loss(spectra_inputs, data_samples,
                                                self.train_cfg)
            spectra_losses.update(add_prefix(spectra_loss_aux, 'aux'))
        
        building_losses = dict()
        if isinstance(self.auxiliary_head3, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head3):
                building_loss_aux = aux_head.loss(building_inputs, data_samples, self.train_cfg)
                building_losses.update(add_prefix(building_loss_aux, f'aux_{idx}'))
        else:
            building_loss_aux = self.auxiliary_head3.loss(building_inputs, data_samples,
                                                self.train_cfg)
            building_losses.update(add_prefix(building_loss_aux, 'aux'))

        return img_losses, spectra_losses, building_losses

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
        inputs1, inputs2, inputs3 = torch.split(inputs, [3, 4, 2], dim=1)
        x1, x2, x3 = self.extract_feat(inputs1, inputs2, inputs3)

        losses = dict()

        img_loss_decode, spectra_loss_decode, building_loss_decode = self._decode_head_forward_train(x1, x2, x3, data_samples)
        # 加权平均
        loss_decode = dict()
        for key in spectra_loss_decode.keys():
            loss_decode[key] = img_loss_decode[key] / 3 + spectra_loss_decode[key] / 3 + building_loss_decode[key] / 3
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            img_loss_aux, spectra_loss_aux, building_loss_aux = self._auxiliary_head_forward_train(x1, x2, x3, data_samples)
            # 加权平均
            loss_aux = dict()
            for key in spectra_loss_aux.keys():
                loss_aux[key] = img_loss_aux[key] / 3 + spectra_loss_aux[key] / 3 + building_loss_aux[key] / 3
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
        # 输入图像是9波段，前3波段为真彩色，中间4波段为光谱指标，后面2波段为建筑指标
        # 将输入图像进行切分
        # inputs1, inputs2, inputs3 = torch.split(inputs, [self.backbone1.in_channels, self.backbone2.in_channels, self.backbone3.in_channels], dim=1)
        inputs1, inputs2, inputs3 = torch.split(inputs, [3, 4, 2], dim=1)

        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        img_seg_logits, spectra_seg_logits, building_seg_logits = self.inference(inputs1, inputs2, inputs3, batch_img_metas)
        # 加权平均
        weight = torch.full(spectra_seg_logits.shape, 1/3).cuda()
        seg_logits = img_seg_logits * weight + spectra_seg_logits * weight + building_seg_logits * weight

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
        inputs1, inputs2, inputs3 = torch.split(inputs, [3, 4, 2], dim=1)
        x1, x2, x3 = self.extract_feat(inputs1, inputs2, inputs3)
        return self.decode_head1.forward(x1), self.decode_head2.forward(x2), self.decode_head3.forward(x3)

    def slide_inference(self, img_inputs: Tensor, spectra_inputs: Tensor, building_inputs: Tensor,
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
        batch_size, _, h_img, w_img = img_inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        img_preds = img_inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        spectra_preds = spectra_inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        building_preds = building_inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        img_count_mat = img_inputs.new_zeros((batch_size, 1, h_img, w_img))
        spectra_count_mat = spectra_inputs.new_zeros((batch_size, 1, h_img, w_img))
        building_count_mat = building_inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                img_crop_img = img_inputs[:, :, y1:y2, x1:x2]
                spectra_crop_img = img_inputs[:, :, y1:y2, x1:x2]
                building_crop_img = img_inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = img_crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                img_crop_seg_logit, spectra_crop_seg_logit, building_crop_seg_logit = self.encode_decode(img_crop_img, spectra_crop_img, building_crop_img, batch_img_metas)
                img_preds += F.pad(img_crop_seg_logit,
                               (int(x1), int(img_preds.shape[3] - x2), int(y1),
                                int(img_preds.shape[2] - y2)))
                spectra_preds += F.pad(spectra_crop_seg_logit,
                               (int(x1), int(spectra_preds.shape[3] - x2), int(y1),
                                int(spectra_preds.shape[2] - y2)))
                building_preds += F.pad(building_crop_seg_logit,
                               (int(x1), int(building_preds.shape[3] - x2), int(y1),
                                int(building_preds.shape[2] - y2)))

                img_count_mat[:, :, y1:y2, x1:x2] += 1
                spectra_count_mat[:, :, y1:y2, x1:x2] += 1
                building_count_mat[:, :, y1:y2, x1:x2] += 1
        assert (img_count_mat == 0).sum() == 0
        img_seg_logits = img_preds / img_count_mat
        spectra_seg_logits = spectra_preds / spectra_count_mat
        building_seg_logits = building_preds / building_count_mat

        return img_seg_logits, spectra_seg_logits, building_seg_logits

    def whole_inference(self, img_inputs: Tensor, spectra_inputs: Tensor, building_inputs: Tensor,
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

        img_seg_logits, spectra_seg_logits, building_seg_logits = self.encode_decode(img_inputs, spectra_inputs, building_inputs, batch_img_metas)

        return img_seg_logits, spectra_seg_logits, building_seg_logits

    def inference(self, img_inputs: Tensor, spectra_inputs: Tensor, building_inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
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
            img_seg_logit, spectra_seg_logit, building_seg_logit = self.slide_inference(img_inputs, spectra_inputs, building_inputs, batch_img_metas)
        else:
            img_seg_logit, spectra_seg_logit, building_seg_logit = self.whole_inference(img_inputs, spectra_inputs, building_inputs, batch_img_metas)

        return img_seg_logit, spectra_seg_logit, building_seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale

        # 输入图像是9波段，前3波段为真彩色，中间4波段为光谱指标，后面2波段为建筑指标
        # 将输入图像进行切分
        inputs1, inputs2, inputs3 = torch.split(inputs, [3, 4, 2], dim=1)

        # to save memory, we get augmented seg logit inplace
        img_seg_logit, spectra_seg_logit, building_seg_logit = self.inference(inputs1[0], inputs2[0], inputs3[0], batch_img_metas[0], rescale)
        # 加权平均
        weight = torch.full(spectra_seg_logit.shape, 1/3).cuda()
        seg_logit = img_seg_logit * weight + spectra_seg_logit * weight + building_seg_logit * weight

        for i in range(1, len(inputs)):
            img_cur_seg_logit, spectra_cur_seg_logit, building_cur_seg_logit = self.inference(inputs1[i], inputs2[i], inputs3[i], batch_img_metas[i],
                                           rescale)
            cur_seg_logit = img_cur_seg_logit * weight + spectra_cur_seg_logit * weight + building_cur_seg_logit * weight
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
