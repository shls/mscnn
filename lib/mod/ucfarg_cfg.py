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
__C.TRAIN.IMS_PER_BATCH = 64
__C.TRAIN.END2END_IMS_PER_BATCH = 8
__C.TRAIN.SPATIAL_CHANNELS = 3
__C.TRAIN.MIX_CHANNELS = 4
__C.TRAIN.TEMPORAL_CHANNELS = 1
__C.TRAIN.LIST_FILE = "/home/ls/dataset/mscnn/data/random_trainval.list"
__C.TRAIN.TRAINLMDB = "/mnt/ssd/ls/dataset/mscnn/trainlmdb"
__C.TRAIN.DATA_EXTENSION = '.npy'
__C.TRAIN.LABEL_EXTENSION = '.txt'
__C.TRAIN.BBOXES_EXTENSION = '.txt'
__C.TRAIN.DATA_ROOT = '/home/ls/dataset/mscnn/data'
__C.TRAIN.LABEL_ROOT = '/home/ls/dataset/mscnn/annotations'
__C.TRAIN.BBOXES_ROOT = '/home/ls/dataset/mscnn/bbox'
__C.TRAIN.SPATIAL_PREFEATURE_ROOT = '/mnt/hdd2/ls/mscnn/spatial_data/roi_pool_spatial_conv4_3'
__C.TRAIN.SPATIAL_PREFEATURE_WIDTH = 64
__C.TRAIN.SPATIAL_PREFEATURE_HEIGHT = 64
__C.TRAIN.SPATIAL_PREFEATURE_CHANNELS = 512
__C.TRAIN.TEMPORAL_PREFEATURE_ROOT = '/home/ls/dataset/roi_pool_temporal_raw'
__C.TRAIN.TEMPORAL_PREFEATURE_WIDTH = 64
__C.TRAIN.TEMPORAL_PREFEATURE_HEIGHT = 64
__C.TRAIN.TEMPORAL_PREFEATURE_CHANNELS = 1
__C.TRAIN.LABEL_PREFEATURE_ROOT = '/home/ls/dataset/label_reshape'
__C.TRAIN.ORG_W = 960
__C.TRAIN.ORG_H = 540
__C.TRAIN.TARGET_W = 960
__C.TRAIN.TARGET_H = 720
__C.TRAIN.RATIOS = (1, 1.33)
__C.TRAIN.MEAN_3 = [104, 117, 123]
__C.TRAIN.MEAN_4 = [115, 115, 115, 115]
__C.TRAIN.MEAN_1 = [115,]
__C.TRAIN.NMS_THRESH = 0.3
__C.TRAIN.NMS_CONFI_THRESH = 0.9
__C.TRAIN.ENLARGE_SPATIAL = True

__C.VAL = edict()
__C.VAL.VALLMDB = "/mnt/ssd/ls/dataset/mscnn/valmdb"
__C.VAL.DATA_ROOT = '/home/ls/dataset/mscnn/data'
__C.VAL.LABEL_ROOT = '/home/ls/dataset/mscnn/annotations'
__C.VAL.LIST_FILE = "/home/ls/dataset/mscnn/data/val.list"
__C.VAL.MEAN_3 = [104, 117, 123]
__C.VAL.MEAN_4 = [115, 115, 115, 115]

__C.TEST = edict()
__C.TEST.IMS_PER_BATCH = 1024
__C.TEST.END2END_IMS_PER_BATCH = 10
__C.TEST.SPATIAL_CHANNELS = 3
__C.TEST.MIX_CHANNELS = 4
__C.TEST.TEMPORAL_CHANNELS = 1
__C.TEST.LIST_FILE = "/home/ls/dataset/mscnn/data/random_test.list"
__C.TEST.TESTLMDB = "/mnt/ssd/ls/dataset/mscnn/testlmdb"
__C.TEST.DATA_EXTENSION = '.npy'
__C.TEST.LABEL_EXTENSION = '.txt'
__C.TEST.BBOXES_EXTENSION = '.txt'
__C.TEST.DATA_ROOT = '/home/ls/dataset/mscnn/data'
__C.TEST.LABEL_ROOT = '/home/ls/dataset/mscnn/annotations'
__C.TEST.BBOXES_ROOT = '/home/ls/dataset/mscnn/bbox'
__C.TEST.SPATIAL_PREFEATURE_ROOT = '/mnt/hdd2/ls/mscnn/spatial_data/roi_pool_spatial_conv4_3'
__C.TEST.SPATIAL_PREFEATURE_WIDTH = 64
__C.TEST.SPATIAL_PREFEATURE_HEIGHT = 64
__C.TEST.SPATIAL_PREFEATURE_CHANNELS = 512
__C.TEST.TEMPORAL_PREFEATURE_ROOT = '/home/ls/dataset/roi_pool_temporal_raw'
__C.TEST.TEMPORAL_PREFEATURE_WIDTH = 64
__C.TEST.TEMPORAL_PREFEATURE_HEIGHT = 64
__C.TEST.TEMPORAL_PREFEATURE_CHANNELS = 1
__C.TEST.LABEL_PREFEATURE_ROOT = '/home/ls/dataset/label_reshape'
__C.TEST.ORG_W = 960
__C.TEST.ORG_H = 540
__C.TEST.TARGET_W = 960
__C.TEST.TARGET_H = 720
__C.TEST.RATIOS = (1, 1.33)
__C.TEST.MEAN_3 = [104, 117, 123]
__C.TEST.MEAN_4 = [115, 115, 115, 115]
__C.TEST.MEAN_1 = [115,]
__C.TEST.NMS_THRESH = 0.3
__C.TEST.NMS_CONFI_THRESH = 0.9
__C.TEST.ENLARGE_SPATIAL = True
# Common settings
__C.GPU_ID = 1
