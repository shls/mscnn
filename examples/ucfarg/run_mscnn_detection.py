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
import caffe
from nms.gpu_nms import gpu_nms

CALTECH_DATA_PATH = "/root/caltech/data/"
IMG_PATH = os.path.join(CALTECH_DATA_PATH + "images")


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a MSCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--net', dest='prototxt',
                        help='prototxt file defining the network',
                        default='/home/ls/mscnn/examples/caltech/mscnn-7s-720-pretrained/mscnn_deploy.prototxt', type=str)
    parser.add_argument('--weights', dest='caffemodel',
                        help='model to test',
                        default='/home/ls/mscnn/examples/caltech/mscnn-7s-720-pretrained/mscnn_caltech_train_2nd_iter_20000.caffemodel'\
                        , type=str)
    parser.add_argument('--do_bb_norm', dest='do_bb_norm',help="Whether to denormalize the box with std or means.\
    Author's pretrained model does not need this. ",
                default=True , type=bool)
    parser.add_argument('--height', dest='height',help="Decide the resizing height of input model and images",
                default=720 , type=int)
    parser.add_argument('--detection', dest='dt_name',  help='model to test', default='detection_1', type=str)
    parser.add_argument('--video_file', dest='video_name',  help='video to test', default='', type=str)
    parser.add_argument('--threshold', dest='threshold', help='threshold for boxes', default=0.9, type=float)
    parser.add_argument('--recursive', dest='recursive', help='recursively testing', default=False, type=bool)
    parser.add_argument('--folder', dest='root_folder', help='root folder for recursively test', default='', type=str)
    parser.add_argument('--filetype', dest='filetype', help='file type(video/img)', default='img',type=str)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
   

def filter_proposals(proposals, threshold=-10):
    #Bug 1 Fixed
    keeps = (proposals[:, -1] >= threshold) & (proposals[:, 2] != 0) & (proposals[:, 3] != 0)
    return keeps


def im_normalize(im, target_size, mu=[104, 117, 123] ):
    n_im = cv2.resize(im, target_size).astype(np.float32)
    
    #Substracts mu from testing-BGR image
    n_im -= mu
    #print(im.shape)
    n_im = np.swapaxes(n_im, 1,2)
    n_im = np.swapaxes(n_im, 0,1)
    n_im = np.array([n_im])
    #print(n_im.shape)
    #print(n_im.shape)
    return n_im


def bbox_denormalize(bbox_pred, proposals, ratios, orgW, orgH):
    
    bbox_means = [0, 0, 0, 0]
    bbox_stds = [0.1, 0.1, 0.2, 0.2]

    if args.do_bb_norm:
        bbox_pred *= bbox_stds 
        bbox_pred += bbox_means

    ctr_x = proposals[:,0]+0.5*proposals[:,2]
    ctr_y = proposals[:,1]+0.5*proposals[:,3]

    tx = bbox_pred[:,0] *proposals[:,2] + ctr_x
    ty = bbox_pred[:,1] *proposals[:,3] + ctr_y

    tw = proposals[:,2] * np.exp(bbox_pred[:,2])
    th = proposals[:,3] * np.exp(bbox_pred[:,3])

    #Fix Bug 2
    tx -= tw/2 
    ty -= th/2
    tx /= ratios[0] 
    tw /= ratios[0]
    ty /= ratios[1] 
    th /= ratios[1]

    tx[tx < 0] = 0
    ty[ty < 0] = 0
    #Fix Bug 3
    tw[tw > (orgW - tx)] = (orgW - tx[tw > (orgW - tx)])
    th[th > (orgH - ty)] = (orgH - ty[th > (orgH - ty)])
    new_boxes = np.hstack((tx[:, None], ty[:, None], tw[:, None], th[:, None])).astype(np.float32).reshape((-1, 4)) #suspecious
    return new_boxes


def get_confidence(cls_pred):
    exp_score = np.exp(cls_pred)
    sum_exp_score = np.sum(exp_score, 1)
    confidence = exp_score[:, 1] / sum_exp_score
    
    return confidence

#mu is the mean of BGR 
# im_dect use file path
# def im_detect(net, file_path, target_size= (960, 720)):

#     im = cv2.imread(file_path)

# im_detect use im
def im_detect(net, im, target_size= (960, 720)):
    orgH, orgW, _ = im.shape
    ratios = (target_size[0]/orgW, (target_size[1]/orgH ))
    im = im_normalize(im, target_size)
    
    #Feedforward
    net.blobs['data'].data[...] = im 
    output = net.forward()
    
    bbox_pred = output['bbox_pred']
    proposals = output['proposals_score'].reshape((-1,6))[:,1:]  #suspecious
    
    proposals[:,2] -=   proposals[:,0]
    proposals[:,3] -=   proposals[:,1]
    cls_pred = output['cls_pred']
    
    
    keeps = filter_proposals(proposals)
    bbox_pred =  bbox_pred[keeps]
    cls_pred = cls_pred[keeps]
    proposals = proposals[keeps]
    
    pedestrian_boxes = bbox_pred[:,4:8]
    boxes = bbox_denormalize(pedestrian_boxes, proposals, ratios, orgW, orgH)

    #Denormalize the confidence 
    
    confidence = get_confidence(cls_pred)
    return confidence, boxes

def nms(dets, thresh):
    
    if dets.shape[0] == 0:
        return []
    new_dets = np.copy(dets)
    new_dets[:,2] += new_dets[:,0]
    new_dets[:,3] += new_dets[:,1]
   
    return gpu_nms(new_dets, thresh, device_id=GPU_ID)

def video_prediction(net, video_name, thresh):
    vidcap = cv2.VideoCapture(video_name)
    success,im = vidcap.read()
    print video_name
    index = 0
    while success: 
        filename = os.path.splitext(os.path.basename(video_name))[0]
        folder = os.path.join("/home/ls/dataset/fe/", filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        savepath =folder + "/" + filename + "." + str(index).zfill(4) + ".txt" 
        # print savepath
	index += 1
        # visual_output(net, im, thresh)
        bbox2file(net, im, thresh, savepath)
        success,im = vidcap.read()

def image_prediction(net, imagename, thresh):
    im = cv2.imread(imagename)
    visual_output(net, im, thresh)


def bbox2file(net, im, thresh, savepath):
    confidence, bboxes = im_detect(net, im)
    dets = np.hstack((bboxes,confidence[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, 0.3)
    dets_nms = dets[keep, :]
    inds = np.where(dets_nms[:, -1] >= thresh)[0]
    file = open(savepath,"w")
    for i in inds:
        bbox = dets_nms[i, :4]
        score = dets_nms[i, -1]
        if score >=thresh:
            x = bbox[0]
            y = bbox[1]
            width = bbox[2]
            height  = bbox[3]
            line = "%s %s %s %s" %(x, y, x+width, y+width)
            #print line
            file.write(line)
        else:
            print "No object detected"
            line = ""
            #print line
            file.write(line)
    file.close()

def visual_output(net, im, thresh):
    confidence, bboxes = im_detect(net, im)
    dets = np.hstack((bboxes,confidence[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, 0.3)
    dets_nms = dets[keep, :]
    inds = np.where(dets_nms[:, -1] >= thresh)[0]
    for i in inds:
        bbox = dets_nms[i, :4]
        score = dets_nms[i, -1]
        if score >=thresh:
            x = bbox[0]
            y = bbox[1]
            width = bbox[2]
            height  = bbox[3]
            im_bbox = im
            print "%s %s %s %s" %(x, y, x+width, y+width)
            #cv2.rectangle(im_bbox, (x,y), (x+width, y+height), (0,255,0),2)
            #cv2.imshow('detections', im_bbox)
            #cv2.waitKey(1)
        else:
            print "No object detected"



def recursive_prediction(net, root_folder, filetype, thresh):
    for f in sorted(os.listdir(root_folder)):
        f = os.path.join(root_folder, f)
        if (os.path.isfile(f)):
        	if filetype == 'img':
        		image_prediction(net, f, thresh)
        	else:
        		video_prediction(net, f, thresh)
        else:
            print f + "is not existed"
            pass
                
if __name__ == "__main__":
    args = parse_args()
    global GPU_ID
    global DETECTION_NAME
    
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    GPU_ID = args.gpu_id
    threshold = args.threshold
    DETECTION_NAME = args.dt_name
    recursive = args.recursive
    root_folder = args.root_folder
    video_name = args.video_name
    filetype = args.filetype

    if recursive and root_folder =='':
        print "Please specify root folder path"
        sys.exit(1)
    elif not recursive and video_name =='' :
        print "Please specify the video path"
        sys.exit(1)

    print("Loading Network")
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    print("MC-CNN model loaded")
    # Detect the video
    if recursive:
        recursive_prediction(net, root_folder, filetype, threshold)
    else:
        video_prediction(net,args.video_name,threshold)

    # Detect the caltech dataset
    # print("Start Detecting")
    # print(IMG_PATH)
    # write_caltech_results_file(net)                 
