#
#   Darknet yolo-VOC model
#   Copyright EAVISE
#

import os
from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet.network as lnn
import lightnet.data as lnd

__all__ = ['YoloVoc']


class YoloVoc(lnn.Darknet):
    """ yolo-voc.2.0.cfg implementation with pytorch.
    This network uses :class:`~lightnet.network.RegionLoss` as loss function
    and :class:`~lightnet.data.BBoxConverter` as postprocessing function.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        weights_file (str, optional): Path to the saved weights; Default **None**
        input_dim (list, optional): Input dimension for the network; Default **[416,416,3]**
        conf_thresh (Number, optional): Confidence threshold for postprocessing of the boxes; Default **0.25**
        nms_thresh (Number, optional): Non-maxima suppression threshold for postprocessing; Default **0.4**
    """
    def __init__(self, num_classes=20, weights_file=None, input_dim=[416,416,3], conf_thresh=.25, nms_thresh=.4):
        """ Network initialisation """
        super(YoloVoc, self).__init__()

        # Parameters
        self.input_dim = input_dim[:]
        self.num_classes = num_classes
        self.anchors = [1.3221,1.73145, 3.19275,4.00944, 5.05587,8.09892, 9.47112,4.84053, 11.2364,10.0071]
        self.num_anchors = 5

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchLeaky(self.input_dim[2], 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchLeaky(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnn.layer.Conv2dBatchLeaky(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnn.layer.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('10_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnn.layer.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('14_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('15_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('16_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('17_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('20_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('21_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('22_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('23_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('24_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
                ('25_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
            ]),

            # Sequence 2 : input = sequence0
            OrderedDict([
                ('26_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 64, 1, 1, 0)),
                ('27_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('28_convbatch',    lnn.layer.Conv2dBatchLeaky((4*64)+1024, 1024, 3, 1, 1)),
                ('29_conv',         nn.Conv2d(1024, self.num_anchors*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        # Weights
        self.load_weights(weights_file)

        # Loss
        self.loss = lnn.RegionLoss(self) 

        # Postprocessing
        self.postprocess = lnd.BBoxConverter(self, conf_thresh, nms_thresh)

    def _forward(self, x):
        outputs = []
    
        outputs.append(self.layers[0](x))
        outputs.append(self.layers[1](outputs[0]))
        # Route : layers=-9
        outputs.append(self.layers[2](outputs[0]))
        # Route : layers=-1,-4
        out = self.layers[3](torch.cat((outputs[2], outputs[1]), 1))

        return out
