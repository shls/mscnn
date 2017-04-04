import numpy as np
import os
import cv2
from ucfarg_cfg import ucfarg_cfg
from bbox_cal import nms, box_iou, center_distance, comp_bbox, bbox_denormalize, filter_proposals, get_confidence

import caffe, json


class ModDataLayer_alternative(caffe.Layer):
	# """docstring for ModDataLayer"""

	def setup(self, bottom, top):

		self._cur_img = 0
		self._bboxes_buf = []
		self._last_img_id = 0
		self._cur_buf = 0

		if (self.phase == caffe.TRAIN):
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TRAIN.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TRAIN.IMS_PER_BATCH
			self._mix_channels = ucfarg_cfg.TRAIN.MIX_CHANNELS
			self._org_h = ucfarg_cfg.TRAIN.ORG_H
			self._org_w = ucfarg_cfg.TRAIN.ORG_W
			self._data_root = ucfarg_cfg.TRAIN.DATA_ROOT
			self._data_extension = ucfarg_cfg.TRAIN.DATA_EXTENSION
			self._mean_4 = ucfarg_cfg.TRAIN.MEAN_4
			self._label_root = ucfarg_cfg.TRAIN.LABEL_ROOT
			self._label_extension = ucfarg_cfg.TRAIN.LABEL_EXTENSION
			self._bboxes_root = ucfarg_cfg.TRAIN.BBOXES_ROOT 
			self._bboxes_extension = ucfarg_cfg.TRAIN.BBOXES_EXTENSION
		else:
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TEST.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TEST.IMS_PER_BATCH
			self._mix_channels = ucfarg_cfg.TEST.MIX_CHANNELS
			self._org_h = ucfarg_cfg.TEST.ORG_H
			self._org_w = ucfarg_cfg.TEST.ORG_W
			self._data_root = ucfarg_cfg.TEST.DATA_ROOT
			self._data_extension = ucfarg_cfg.TEST.DATA_EXTENSION
			self._mean_4 = ucfarg_cfg.TEST.MEAN_4
			self._label_root = ucfarg_cfg.TEST.LABEL_ROOT
			self._label_extension = ucfarg_cfg.TEST.LABEL_EXTENSION
			self._bboxes_root = ucfarg_cfg.TEST.BBOXES_ROOT 
			self._bboxes_extension = ucfarg_cfg.TEST.BBOXES_EXTENSION

                top[0].reshape(self._imgs_per_batch,self._mix_channels, self._org_h, self._org_w)
                top[1].reshape(self._imgs_per_batch,5)
                top[2].reshape(self._imgs_per_batch,1)

	def forward(self, bottom, top):

		# Init result blob
		blob_mix_im = np.zeros((self._imgs_per_batch, self._mix_channels, self._org_h, self._org_w), dtype=np.float32)
		blob_rois = np.zeros((1,5), dtype=np.float32)
		blob_label = np.zeros((1,1,1,1), dtype=np.int)

		for batch_index in xrange(self._imgs_per_batch):

			if self._cur_img == len(self._indexlist):
				self._cur_img = 0
			# Get mix img and feed it into blob
			img_basename = self._indexlist[self._cur_img]
			mix_im = np.asarray(np.load(os.path.join(self._data_root, img_basename + self._data_extension)))
			mix_im = mix_im.astype(np.float32)
			mix_im -= self._mean_4
			mix_im = mix_im.transpose((2,0,1))
			blob_mix_im[batch_index, :, :, :] = mix_im

			if mix_im.shape[0] != 4:
				print "image shape mismatch by Ls"
				raise
			# Get annotation 
			label_anno = os.path.join(self._label_root, img_basename + self._label_extension)
			assert os.path.exists(label_anno), 'Path does not exist: {}'.format(label_anno)
			with open(label_anno) as f:
				lable_anno_data = f.readline()
			label = int(lable_anno_data.split()[0])
			img_id = int(lable_anno_data.split()[1])
			# Get boxes
			bbox_anno = os.path.join(self._bboxes_root, os.path.dirname(img_basename), os.path.basename(img_basename)[4:] + self._bboxes_extension)
			assert os.path.exists(bbox_anno), 'Path does not exist: {}'.format(bbox_anno)
			with open(bbox_anno) as f:
				bboxes = []
				for line in f:
					line = line.split()
					if line:
						line = [float(i) for i in line]
						bboxes.append(line)

			if self._last_img_id != img_id:
				self._cur_buf = 0
				self._bboxes_buf = []
				self._last_img_id = img_id
				if len(bboxes) != 0:
					self._bboxes_buf.append(bboxes)
					self._cur_buf += 1

				if len(blob_rois) == 1 and np.count_nonzero(blob_rois) == 0:
					blob_rois[0,:] = np.array([batch_index,0,0,128,128])
					blob_label[0,0,0,0] = 0
				else:
					blob_rois = np.concatenate((blob_rois, np.array([[batch_index,0,0,128,128],])))
					blob_label = np.concatenate((blob_label,np.array([[[[0]]]])))
			else:
				if len(bboxes) != 0:
					if len(self._bboxes_buf) != 30:
						self._bboxes_buf.append(bboxes)
						self._cur_buf += 1
						blob_rois = np.concatenate((blob_rois, np.array([[batch_index,0,0,128,128],])))
						blob_label = np.concatenate((blob_label,np.array([[[[0]]]])))
					else:
						for i in reversed(xrange(self._cur_buf)):
							bboxes = comp_bbox(np.asarray(bboxes), np.asarray(self._bboxes_buf[i]))
						for i in range(29,self._cur_buf-1,-1):
							bboxes = comp_bbox(np.asarray(bboxes), np.asarray(self._bboxes_buf[i]))

						if self._cur_buf == 30:
							self._cur_buf = 0
							self._bboxes_buf[self._cur_buf] = bboxes[:]
							self._cur_buf += 1
						else:
							self._bboxes_buf[self._cur_buf] = bboxes[:]
							self._cur_buf += 1

						#Compensate for batch index
						bboxes = np.insert(bboxes,0,batch_index,axis=1)
						
						blob_rois = np.concatenate((blob_rois, bboxes))
						blob_label = np.concatenate((blob_label, np.full((len(bboxes),1,1,1), label)))

				else:
					if len(self._bboxes_buf) == 30:
						if self._cur_buf == 30:
							self._cur_buf = 0
							self._bboxes_buf[self._cur_buf] = self._bboxes_buf[29][:]
							self._cur_buf += 1
						else:
							self._bboxes_buf[self._cur_buf] = self._bboxes_buf[self._cur_buf - 1][:]
							self._cur_buf += 1
					elif len(self._bboxes_buf) != 0:
						self._bboxes_buf.append(self._bboxes_buf[self._cur_buf - 1])
						self._cur_buf += 1
					else: # not the first image, but buf is still empty, the new bbox is empty
						pass
					blob_rois = np.concatenate((blob_rois, np.array([[batch_index,0,0,128,128],])))
					blob_label = np.concatenate((blob_label,np.array([[[[0]]]])))

			self._cur_img += 1

		top[0].reshape(*(blob_mix_im.shape))
		top[0].data[...] = blob_mix_im.astype(np.float32, copy=False)

		top[1].reshape(*(blob_rois.shape))
		top[1].data[...] = blob_rois.astype(np.float32, copy=False)

		top[2].reshape(*(blob_label.shape))
		top[2].data[...] = blob_label.astype(np.float32, copy=False)

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass
