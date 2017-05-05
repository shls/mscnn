import numpy as np
import os
import cv2
from ucfarg_cfg import ucfarg_cfg
from bbox_cal import nms, box_iou, center_distance, comp_bbox, bbox_denormalize, filter_proposals, get_confidence
import lmdb
import mod_pb2
import timer
import caffe, json


class ModDataLayer_alter_s1_lmdb(caffe.Layer):
	# """docstring for ModDataLayer"""

	def setup(self, bottom, top):

		if (self.phase == caffe.TRAIN):
			# Setup ModDataLayer
			self._imgs_per_batch = ucfarg_cfg.TRAIN.IMS_PER_BATCH
			self._lmdb = lmdb.open(ucfarg_cfg.TRAIN.TRAINLMDB)
			self._mix_channels = ucfarg_cfg.TRAIN.MIX_CHANNELS
			self._org_h = ucfarg_cfg.TRAIN.ORG_H
			self._org_w = ucfarg_cfg.TRAIN.ORG_W
			self._mean_4 = ucfarg_cfg.TRAIN.MEAN_4
		else:
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TEST.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TEST.IMS_PER_BATCH
			self._lmdb = lmdb.open(ucfarg_cfg.TEST.TESTLMDB)
			self._mix_channels = ucfarg_cfg.TEST.MIX_CHANNELS
			self._org_h = ucfarg_cfg.TEST.ORG_H
			self._org_w = ucfarg_cfg.TEST.ORG_W
			self._mean_4 = ucfarg_cfg.TEST.MEAN_4

		self._cur_img = 0
		self._lmdb_length = int(self._lmdb.stat()['entries'])
		self._bboxes_buf = []
		self._last_img_id = 0
		self._cur_buf = 0

		top[0].reshape(self._imgs_per_batch,self._mix_channels, self._org_h, self._org_w)
		top[1].reshape(self._imgs_per_batch,5)
		top[2].reshape(self._imgs_per_batch,1)

	def forward(self, bottom, top):

		start = time.time()

		# Init result blob
		blob_mix_im = np.zeros((self._imgs_per_batch, self._mix_channels, self._org_h, self._org_w), dtype=np.float32)
		blob_rois = np.zeros((1,5), dtype=np.float32)
		blob_label = np.zeros((1,1,1,1), dtype=np.int)
		lmdb_begin = self._lmdb.begin()

		for batch_index in xrange(self._imgs_per_batch):

			if self._cur_img == self._lmdb_length:
				self._cur_img = 0

			raw_datum = lmdb_begin.get('{:08}'.format(self._cur_img))
			datum = mod_pb2.MOD()
			datum.ParseFromString(raw_datum)

			# Get mix img and feed it into blob
			mix_im = np.fromstring(datum.data, dtype=np.float32)
			mix_im -= self._mean_4
			mix_im = mix_im.transpose((2,0,1))
			blob_mix_im[batch_index, :, :, :] = mix_im

			if mix_im.shape[0] != 4:
				print "image shape mismatch by Ls"
				raise

			# Get annotation 
			label = int(datum.label)
			img_id = int(datum.clip_id)

			# Get boxes
			bboxes = []
			if int(datum.bboxes) > 0:
				cache_bbox = [float(datum.bbox1_tl), float(datum.bbox1_tr), float(datum.bbox1_dl), float(datum.bbox1_dr)]
				bboxes.append(cache_bbox)
				if int(datum.bboxes) > 1:
					cache_bbox = [float(datum.bbox2_tl), float(datum.bbox2_tr), float(datum.bbox2_dl), float(datum.bbox2_dr)]
					bboxes.append(cache_bbox)
			else:
				pass

			end = time.time()
			print "Time eclapsed for data loading: ", end - start, " s" 
			start = end


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
						if len(blob_rois) == 1 and np.count_nonzero(blob_rois) == 0:
							blob_rois[0,:] = np.array([batch_index,0,0,128,128])
							blob_label[0,0,0,0] = 0
						else:
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

						if len(blob_rois) == 1 and np.count_nonzero(blob_rois) == 0:
							blob_rois[0,:] = bboxes[0][:]
							blob_label[0,0,0,0] = np.full((1,1,1,1), label)
							if len(bboxes) > 1:
								blob_rois = np.concatenate((blob_rois, bboxes[1:]))
								blob_label = np.concatenate((blob_label, np.full((len(bboxes)-1,1,1,1), label)))
						else:
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
					if len(blob_rois) == 1 and np.count_nonzero(blob_rois) == 0:
						blob_rois[0,:] = np.array([batch_index,0,0,128,128])
						blob_label[0,0,0,0] = 0
					else:
						blob_rois = np.concatenate((blob_rois, np.array([[batch_index,0,0,128,128],])))
						blob_label = np.concatenate((blob_label,np.array([[[[0]]]])))

			self._cur_img += 1
		
		end = time.time()
		print "Time eclapsed for bbox caculation: ", end - start, " s" 
		
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
