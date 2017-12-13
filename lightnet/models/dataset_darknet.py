#
#   Darknet dataset
#   Copyright EAVISE
#

import os
from PIL import Image
from torchvision import transforms as tf

import lightnet.data as lnd

__all__ = ['DarknetData']


class DarknetData(lnd.BramboxData):
    """ Dataset that works with darknet files and performs the same data augmentations.
    If train is False, you must use this dataset with the :meth:`~lightnet.data.bbb_collate` function in a dataloader.
        
    Args:
        data_file (str): File containing path to image files (relative from where command is run)
        network (lightnet.network.Darknet): Network that will be used with dataset (needed for network input dimension)
        train (Boolean, optional): Indicates whether to return the annotation or a tensor; Default **True**
        augment (Boolean, optional): Whether or not you want data augmentation; Default **True**
        jitter (Number [0-1], optional): Determines random crop sizes; Default **0.2**
        flip (Number [0-1], optional): Determines whether image will be flipped; Default **0.5**
        hue (Number, optional): Determines hue shift; Default **0.1**
        saturation (Number, optional): Determines saturation shift; Default **1.5**
        value (Number, optional): Determines value (exposure) shift; Default **1.5**
        class_label_map (list, optional): class label map to convert class names to an index; Default **None**
    """
    def __init__(self, data_file, network, train=True, augment=True, jitter=.2, flip=.5, hue=.1, saturation=1.5, value=1.5, class_label_map=None):
        with open(data_file, 'r') as f:
            self.img_paths = f.read().splitlines()

        # Prepare variables for brambox init
        anno_format = 'anno_darknet'
        self.anno_paths = [os.path.splitext(p)[0]+'.txt' for p in self.img_paths]
        identify = lambda name : self.img_paths[self.anno_paths.index(name)]
        
        lb  = lnd.Letterbox(network)
        rf  = lnd.RandomFlip(flip)
        rc  = lnd.RandomCrop(jitter, True)
        hsv = lnd.HSVShift(hue, saturation, value)
        at  = lnd.AnnoToTensor(network)
        it  = tf.ToTensor()
        if augment:
            img_tf = tf.Compose([hsv, rc, rf, lb, it])
            anno_tf = tf.Compose([rc, rf, lb])
        else:
            img_tf = tf.Compose([lb, it])
            anno_tf = tf.Compose([lb])
        if train:
            anno_tf.transforms.append(at)
        
        first_img = Image.open(self.img_paths[0])
        w, h = first_img.size
        kwargs = { 'image_width': w, 'image_height':h, 'class_label_map': class_label_map }

        super(DarknetData, self).__init__(anno_format, self.anno_paths, identify, img_tf, anno_tf, **kwargs)

        # Memory optimisation: set AnnoToTensor maximum
        self.max_anno = max([len(anno) for _,anno in self.annos.items()])
        if train:
            at.max = self.max_anno
