import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import sdha_cfg
ucfarg_cfg = __C

__C.TRAIN = edict()

__C.TRAIN.IMS_PER_BATCH = 1
__C.TRAIN.SPATIAL_CHANNELS = 3
__C.TRAIN.MIX_CHANNELS = 4
__C.TRAIN.WIDTH = 960
__C.TRAIN.HEIGHT = 720
__C.TRAIN.LIST_FILE = "/home/ls/dataset/mscnn/data/train.list"
__C.TRAIN.DATA_EXTENSION = '.npy'
__C.TRAIN.LABEL_EXTENSION = '.txt'
__C.TRAIN.DATA_ROOT = '/home/ls/dataset/mscnn/data'
__C.TRAIN.LABEL_ROOT = '/home/ls/dataset/mscnn/annotations'
__C.TRAIN.ORG_W = 960
__C.TRAIN.ORG_H = 540
__C.TRAIN.TARGET_W = 960
__C.TRAIN.TARGET_H = 720
__C.TRAIN.RATIOS = (1, 0.75)
__C.TRAIN.MEAN_3 = [104, 117, 123]
__C.TRAIN.MEAN_4 = [115, 115, 115, 115]
__C.TRAIN.NMS_THRESH = 0.3
__C.TRAIN.NMS_CONFI_THRESH = 0.9

__C.VAL = edict()

__C.VAL.DATA_ROOT = '/home/ls/dataset/mscnn/data'
__C.VAL.LABEL_ROOT = '/home/ls/dataset/mscnn/annotations'
__C.VAL.LIST_FILE = "/home/ls/dataset/mscnn/data/val.list"
__C.VAL.MEAN_3 = [104, 117, 123]
__C.VAL.MEAN_4 = [115, 115, 115, 115]


__C.GPU_ID = 1

__C.category = 2
__C.two_category = ('__background__', 'potential_event')
__C.seven_category = ('__background__', 'Hand Shaking', 'Hugging', 'Kicking', 'Pointing', 'Punching', 'Pushing')

__C.channels = 1
__C.stream_name = 'temporal'
__C.subdataset = 'mhi10'

__C.device_name = 'GTX780'
__C.GTX780_root = '/mnt/naruto/data_sets/sdha/rcnn'
__C.GTX980_root = '/extra/ls/sdha/rcnn'
