# Modified from "https://raw.githubusercontent.com/GBJim/mscnn/master/examples/caltech/run_mscnn_detection.py"
# Modified by shls

from __future__ import division
import os
import math
import numpy as np
import json
from os import listdir
from os.path import isfile, join
import sys
import glob
import cv2
import argparse
import re
import time
from scipy.misc import imread

# set caffe root and lib   
caffe_root = '/home/ls/mscnn/'
sys.path.insert(0, caffe_root + "install/python")
sys.path.insert(0, caffe_root + "lib")
sys.path.insert(0, caffe_root + "lib/mod")
import caffe
from nms.gpu_nms import gpu_nms
from ucfarg_cfg import ucfarg_cfg


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a MSCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=1, type=int)
    parser.add_argument('--net', dest='prototxt',
                        help='prototxt file defining the network',
                        default='/home/ls/mscnn/examples/ucfarg/feature_extraction/middle_feature/save_feature.prototxt', type=str)
    parser.add_argument('--weights', dest='caffemodel',
                        help='model to test',
                        default='/home/ls/mscnn/examples/ucfarg/feature_extraction/mscnn_caltech_train_2nd_iter_20000.caffemodel'\
                        , type=str)
    parser.add_argument('--bb_norm', dest='do_bb_norm',help="Whether to denormalize the box with std or means.\
    Author's pretrained model does not need this. ",
                action='store_true')
    parser.add_argument('--no_bb_norm', dest='do_bb_norm',help="Whether to denormalize the box with std or means.\
    Author's pretrained model does not need this. ",
                action='store_false')
    parser.set_defaults(do_bb_norm=False)
    parser.add_argument('--height', dest='height',help="Decide the resizing height of input model and images",
                default=720 , type=int)
    parser.add_argument('--detection', dest='dt_name',  help='model to test', default='detection_1', type=str)
    parser.add_argument('--video_file', dest='video_name',  help='video to test', default='', type=str)
    parser.add_argument('--threshold', dest='threshold', help='threshold for boxes', default=0.9, type=float)
    parser.add_argument('--folder', dest='root_folder', help='root folder for recursively test', default='', type=str)
    parser.add_argument('--filetype', dest='filetype', help='file type(video/img)', default='img',type=str)
    parser.add_argument('--with_recursive', dest='recursive', help='recursively testing', action='store_true')
    parser.add_argument('--no_recursive', dest='recursive', help='recursively testing', action='store_false')
    parser.set_defaults(recursive=False)
    parser.add_argument('--feature', dest='feature', help='extract feature and save it', default='', type=str)

    args = parser.parse_args()
    return args

                
if __name__ == "__main__":
    args = parse_args()
    global GPU_ID
    global DETECTION_NAME
    
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    GPU_ID = args.gpu_id

    print("Loading Network")
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    print("MC-CNN model loaded")
    
    _indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TRAIN.LIST_FILE)]
    _data_root = ucfarg_cfg.TRAIN.DATA_ROOT
    _data_extension = ucfarg_cfg.TRAIN.DATA_EXTENSION
    _target_h = ucfarg_cfg.TRAIN.TARGET_H
    _target_w = ucfarg_cfg.TRAIN.TARGET_W
    _mean_3 = ucfarg_cfg.TRAIN.MEAN_3
    _mean_1 = ucfarg_cfg.TRAIN.MEAN_1
    _label_root = ucfarg_cfg.TRAIN.LABEL_ROOT
    _label_extension = ucfarg_cfg.TRAIN.LABEL_EXTENSION
    _save_path = "/home/ls/dataset"

    for index in _indexlist:

        mix_im = np.asarray(np.load(os.path.join(_data_root, index + _data_extension)))
        spatial_im = np.asarray(mix_im[:, :, 0:-1])
        spatial_im = cv2.resize(spatial_im, (_target_w, _target_h)).astype(np.float32)
        spatial_im -= _mean_3
        spatial_im = spatial_im.transpose((2,0,1))

        temporal_im = np.asarray(mix_im[:,:,-1])
        temporal_im -= _mean_1
        temporal_im = np.expand_dims(temporal, axis=2).astype(np.float32)
        temporal_im = temporal_im.transpose((2,0,1))

        label_file = os.path.join(_label_root, index + _label_extension)
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)

        with open(label_file) as f:
            label_data = f.readline()

        label = int(label_data.split()[0])
        clip_id = int(label_data.split()[1])        
        
        net.blobs['data'].data[...] = spatial_im
        net.blobs['mhi'].data[...] = temporal_im
        net.blobs['label'].data[...] = label
        net.blobs['clip_id'].data[...] = clip_id
        output = net.forward()

        # print "Boxes: ", index, " ", net.blobs["bbox_nms"].data[0]

        # conv_list = ['roi_pool_spatial_conv4_3', 'roi_pool_spatial_conv5_3', 'roi_pool_spatial_conv6_1', 'roi_pool_temporal_raw']
        conv_list = ['roi_pool_temporal_raw']

        for i in xrange(len(conv_list)):
            # print output[conv_list[i]].shape
            if not os.path.exists(os.path.join(_save_path, conv_list[i], os.path.dirname(index))):
                os.makedirs(os.path.join(_save_path, conv_list[i], os.path.dirname(index)))
            np.save(os.path.join(_save_path, conv_list[i], index + ".npy"), output[conv_list[i]])

    print "training set already saved"

    _indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TEST.LIST_FILE)]
    _data_root = ucfarg_cfg.TEST.DATA_ROOT
    _data_extension = ucfarg_cfg.TEST.DATA_EXTENSION
    _target_h = ucfarg_cfg.TEST.TARGET_H
    _target_w = ucfarg_cfg.TEST.TARGET_W
    _mean_3 = ucfarg_cfg.TEST.MEAN_3
    _label_root = ucfarg_cfg.TEST.LABEL_ROOT
    _label_extension = ucfarg_cfg.TEST.LABEL_EXTENSION

    for index in _indexlist:

        mix_im = np.asarray(np.load(os.path.join(_data_root, index + _data_extension)))
        spatial_im = np.asarray(mix_im[:, :, 0:-1])
        spatial_im = cv2.resize(spatial_im, (_target_w, _target_h)).astype(np.float32)
        spatial_im -= _mean_3
        spatial_im = spatial_im.transpose((2,0,1))

        temporal_im = np.asarray(mix_im[:,:,-1])
        temporal_im -= _mean_1
        temporal_im = np.expand_dims(temporal, axis=2).astype(np.float32)
        temporal_im = temporal_im.transpose((2,0,1))

        label_file = os.path.join(_label_root, index + _label_extension)
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)

        with open(label_file) as f:
            label_data = f.readline()

        label = int(label_data.split()[0])
        clip_id = int(label_data.split()[1])        
        
        net.blobs['data'].data[...] = spatial_im
        net.blobs['mhi'].data[...] = temporal_im
        net.blobs['label'].data[...] = label
        net.blobs['clip_id'].data[...] = clip_id
        output = net.forward()

        # conv_list = ['roi_pool_spatial_conv4_3', 'roi_pool_spatial_conv5_3', 'roi_pool_spatial_conv6_1', 'roi_pool_temporal_raw']
        conv_list = ['roi_pool_temporal_raw']

        for i in xrange(len(conv_list)):
            if not os.path.exists(os.path.join(_save_path, conv_list[i], os.path.dirname(index))):
                os.makedirs(os.path.join(_save_path, conv_list[i], os.path.dirname(index)))
            np.save(os.path.join(_save_path, conv_list[i], index + ".npy"), output[conv_list[i]]) 

    print "Testing set already saved"

           





