import numpy as np
import os
import cv2
from ucfarg_cfg import ucfarg_cfg
from bbox_cal import nms, box_iou, center_distance, comp_bbox, bbox_denormalize, filter_proposals, get_confidence

import caffe, json


class ModDataLayer_alternative_s2(caffe.Layer):
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
			self._spatial_channels = ucfarg_cfg.TRAIN.SPATIAL_CHANNELS
			self._temporal_channels = ucfarg_cfg.TRAIN.TEMPORAL_CHANNELS
			self._org_h = ucfarg_cfg.TRAIN.ORG_H
			self._org_w = ucfarg_cfg.TRAIN.ORG_W
			self._data_root = ucfarg_cfg.TRAIN.DATA_ROOT
			self._data_extension = ucfarg_cfg.TRAIN.DATA_EXTENSION
			self._mean_3 = ucfarg_cfg.TRAIN.MEAN_3
			self._mean_1 = ucfarg_cfg.TRAIN.MEAN_1
			self._label_root = ucfarg_cfg.TRAIN.LABEL_ROOT
			self._label_extension = ucfarg_cfg.TRAIN.LABEL_EXTENSION
			self._bboxes_root = ucfarg_cfg.TRAIN.BBOXES_ROOT 
			self._bboxes_extension = ucfarg_cfg.TRAIN.BBOXES_EXTENSION
			self._enlarge_spatial = ucfarg_cfg.TRAIN.ENLARGE_SPATIAL
		else:
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TEST.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TEST.IMS_PER_BATCH
			self._spatial_channels = ucfarg_cfg.TEST.SPATIAL_CHANNELS
			self._temporal_channels = ucfarg_cfg.TEST.TEMPORAL_CHANNELS
			self._org_h = ucfarg_cfg.TEST.ORG_H
			self._org_w = ucfarg_cfg.TEST.ORG_W
			self._data_root = ucfarg_cfg.TEST.DATA_ROOT
			self._data_extension = ucfarg_cfg.TEST.DATA_EXTENSION
			self._mean_3 = ucfarg_cfg.TEST.MEAN_3
			self._mean_1 = ucfarg_cfg.TEST.MEAN_1
			self._label_root = ucfarg_cfg.TEST.LABEL_ROOT
			self._label_extension = ucfarg_cfg.TEST.LABEL_EXTENSION
			self._bboxes_root = ucfarg_cfg.TEST.BBOXES_ROOT 
			self._bboxes_extension = ucfarg_cfg.TEST.BBOXES_EXTENSION
			self._enlarge_spatial = ucfarg_cfg.TEST.ENLARGE_SPATIAL

		top[0].reshape(self._imgs_per_batch, self._spatial_channels, self._org_h, self._org_w)
		top[1].reshape(self._imgs_per_batch, self._temporal_channels, self._org_h, self._org_w)
		top[2].reshape(1,5)
		top[3].reshape(1,5)
		top[4].reshape(1)


	def enlarge_bbox(self, tight_bboxes):
		for i in xrange(len(tight_bboxes)):
			x = tight_bboxes[i][1]
			y = tight_bboxes[i][2]
			width = tight_bboxes[i][3] - tight_bboxes[i][1]
			height = tight_bboxes[i][4] - tight_bboxes[i][2]

			tl_x = x - width if (x-width) > 0 else 0
			tl_y = y - 0.5 * height if (y-0.5*height) > 0 else 0
			rd_x = x + 2 * width if (x+2*width) < self._org_w else self._org_w
			rd_y = y + 1.5 * height if (y+1.5*height) < self._org_h else self._org_h

			tight_bboxes[i][1] = tl_x
			tight_bboxes[i][2] = tl_y
			tight_bboxes[i][3] = rd_x
			tight_bboxes[i][4] = rd_y
		return tight_bboxes

	def forward(self, bottom, top):

		# Init result blob
		blob_spatial_im = np.zeros((self._imgs_per_batch, self._spatial_channels, self._org_h, self._org_w), dtype=np.float32)
		blob_temporal_im = np.zeros((self._imgs_per_batch, self._temporal_channels, self._org_h, self._org_w), dtype=np.float32)
		blob_spatial_rois = np.zeros((1,5), dtype=np.float32)		
		blob_temporal_rois = np.zeros((1,5), dtype=np.float32)
		blob_label = np.zeros((1,1,1,1), dtype=np.int)

		for batch_index in xrange(self._imgs_per_batch):

			if self._cur_img == len(self._indexlist):
				self._cur_img = 0
			
			# Get mix img and feed it into blob
			img_basename = self._indexlist[self._cur_img]
			mix_im = np.asarray(np.load(os.path.join(self._data_root, img_basename + self._data_extension)))
			
			# Split mix images to bgr and mhi
			b,g,r,temporal = cv2.split(mix_im)
			spatial_im = cv2.merge((b,g,r))
			spatial_im = spatial_im.astype(np.float32)
			temporal_im = np.expand_dims(temporal, axis=2).astype(np.float32)

			spatial_im -= self._mean_3
			temporal_im -= self._mean_1
			spatial_im = spatial_im.transpose((2,0,1))
			temporal_im = temporal_im.transpose((2,0,1))

			blob_spatial_im[batch_index, :, :, :] = spatial_im
			blob_temporal_im[batch_index, :, :, :] = temporal_im

			if spatial_im.shape[0] != 3 or temporal_im.shape[0] !=1:
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
			bbox_anno = os.path.join(self._bboxes_root, img_basename + self._bboxes_extension)
			assert os.path.exists(bbox_anno), 'Path does not exist: {}'.format(bbox_anno)
			with open(bbox_anno) as f:
				bboxes = []
				for line in f:
					line = line.split()
					if line:
						line = [float(i) for i in line]
						bboxes.append(line)
			spatial_bboxes = bboxes[:]

			if self._last_img_id != img_id:
				self._cur_buf = 0
				self._bboxes_buf = []
				self._last_img_id = img_id
				if len(bboxes) != 0:
					self._bboxes_buf.append(bboxes)
					self._cur_buf += 1

				if len(blob_temporal_rois) == 1 and np.count_nonzero(blob_temporal_rois) == 0:
					blob_spatial_rois[0,:] = np.array([batch_index,0,0,128,128])
					blob_temporal_rois[0,:] = np.array([batch_index,0,0,128,128])
					blob_label[0,0,0,0] = 0
				else:
					blob_spatial_rois = np.concatenate((blob_spatial_rois, np.array([[batch_index,0,0,128,128],])))
					blob_temporal_rois = np.concatenate((blob_temporal_rois, np.array([[batch_index,0,0,128,128],])))
					blob_label = np.concatenate((blob_label,np.array([[[[0]]]])))
			else:
				if len(bboxes) != 0:
					if len(self._bboxes_buf) != 30:
						self._bboxes_buf.append(bboxes)
						self._cur_buf += 1
						blob_spatial_rois = np.concatenate((blob_spatial_rois, np.array([[batch_index,0,0,128,128],])))
						blob_temporal_rois = np.concatenate((blob_temporal_rois, np.array([[batch_index,0,0,128,128],])))
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
						spatial_bboxes = np.insert(spatial_bboxes,0,batch_index,axis=1)

						#Enlarge spatial
						if self._enlarge_spatial:
							spatial_bboxes = self.enlarge_bbox(spatial_bboxes)
						
						blob_spatial_rois = np.concatenate((blob_spatial_rois,spatial_bboxes))
						blob_temporal_rois = np.concatenate((blob_temporal_rois, bboxes))
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
					blob_spatial_rois = np.concatenate((blob_spatial_rois,np.array([[batch_index,0,0,128,128],])))
					blob_temporal_rois = np.concatenate((blob_temporal_rois, np.array([[batch_index,0,0,128,128],])))
					blob_label = np.concatenate((blob_label,np.array([[[[0]]]])))

			self._cur_img += 1

		top[0].reshape(*(blob_spatial_im.shape))
		top[0].data[...] = blob_spatial_im.astype(np.float32, copy=False)

		top[1].reshape(*(blob_temporal_im.shape))
		top[1].data[...] = blob_temporal_im.astype(np.float32, copy=False)

		top[2].reshape(*(blob_spatial_rois.shape))
		top[2].data[...] = blob_spatial_rois.astype(np.float32, copy=False)

		top[3].reshape(*(blob_temporal_rois.shape))
		top[3].data[...] = blob_temporal_rois.astype(np.float32, copy=False)

		top[4].reshape(*(blob_label.shape))
		top[4].data[...] = blob_label.astype(np.float32, copy=False)

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass
