import numpy as np
import os
import cv2
from ucfarg_cfg import ucfarg_cfg

import caffe, json


class ModDataLayer(caffe.Layer):
	# """docstring for ModDataLayer"""


	def _get_next_batch(self):
	# """Return the blobs to be used for the next minibatch.

	# If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
	# separate process and made available through self._blob_queue.
	# """
		blob_spatial_im = np.zeros((self._imgs_per_batch, self._spatial_channels, self._target_h,self._target_w), dtype=np.float32)
		blob_mix_im = np.zeros((self._imgs_per_batch, self._mix_channels, self._org_h, self._org_w), dtype=np.float32)
		label_blob = np.zeros((0), dtype=np.float32)
		clip_id_blob = np.zeros((0), dtype=np.float32)

		for batch_index in xrange(self._imgs_per_batch):

			if self._cur == len(self._indexlist):
				self._cur = 0

			index = self._indexlist[self._cur]
			mix_im = np.asarray(np.load(os.path.join(self._data_root, index + self._data_extension)))

			spatial_im = np.asarray(mix_im[:, :, 0:-1])
			spatial_im = cv2.resize(spatial_im, (self._target_w, self._target_h)).astype(np.float32)
			spatial_im -= self._mean_3
			spatial_im = spatial_im.transpose((2,0,1))
			blob_spatial_im[batch_index, :, :, :] = spatial_im

			mix_im = mix_im.astype(np.float32)
			mix_im -= self._mean_4
			mix_im = mix_im.transpose((2,0,1))
			blob_mix_im[batch_index, :, :, :] = mix_im

			if mix_im.shape[0] != 4 or spatial_im.shape[0] !=3:
				print "image shape mismatch by Ls"
				raise

			label_file = os.path.join(self._label_root, index + self._label_extension)
			assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)

			with open(label_file) as f:
				label_data = f.readline()

			label = int(label_data.split()[0])
			clip_id = int(label_data.split()[1])

			label_blob = np.hstack((label_blob, label))
			clip_id_blob = np.hstack((clip_id_blob, clip_id))

			self._cur += 1
		
		blobs = {'clip_id': clip_id_blob, 'labels': label_blob, 'mix_data': blob_mix_im, 'spatial_data': blob_spatial_im}
		return blobs

	def setup(self, bottom, top):

		self._cur = 0
		self._name_to_top_map = {}

		if (self.phase == caffe.TRAIN):
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TRAIN.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TRAIN.END2END_IMS_PER_BATCH
			self._spatial_channels = ucfarg_cfg.TRAIN.SPATIAL_CHANNELS
			self._mix_channels = ucfarg_cfg.TRAIN.MIX_CHANNELS
			self._target_h = ucfarg_cfg.TRAIN.TARGET_H
			self._target_w = ucfarg_cfg.TRAIN.TARGET_W
			self._org_h = ucfarg_cfg.TRAIN.ORG_H
			self._org_w = ucfarg_cfg.TRAIN.ORG_W
			self._data_root = ucfarg_cfg.TRAIN.DATA_ROOT
			self._data_extension = ucfarg_cfg.TRAIN.DATA_EXTENSION
			self._mean_3 = ucfarg_cfg.TRAIN.MEAN_3
			self._mean_4 = ucfarg_cfg.TRAIN.MEAN_4
			self._label_root = ucfarg_cfg.TRAIN.LABEL_ROOT
			self._label_extension = ucfarg_cfg.TRAIN.LABEL_EXTENSION

		else:
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TEST.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TEST.END2END_IMS_PER_BATCH
			self._spatial_channels = ucfarg_cfg.TEST.SPATIAL_CHANNELS
			self._mix_channels = ucfarg_cfg.TEST.MIX_CHANNELS
			self._target_h = ucfarg_cfg.TEST.TARGET_H
			self._target_w = ucfarg_cfg.TEST.TARGET_W
			self._org_h = ucfarg_cfg.TEST.ORG_H
			self._org_w = ucfarg_cfg.TEST.ORG_W
			self._data_root = ucfarg_cfg.TEST.DATA_ROOT
			self._data_extension = ucfarg_cfg.TEST.DATA_EXTENSION
			self._mean_3 = ucfarg_cfg.TEST.MEAN_3
			self._mean_4 = ucfarg_cfg.TEST.MEAN_4
			self._label_root = ucfarg_cfg.TEST.LABEL_ROOT
			self._label_extension = ucfarg_cfg.TEST.LABEL_EXTENSION

		idx = 0
		top[idx].reshape(self._imgs_per_batch, self._spatial_channels,
			self._target_h,self._target_w)
		self._name_to_top_map['spatial_data'] = idx
		idx += 1

		top[idx].reshape(self._imgs_per_batch, self._mix_channels,
			self._org_h, self._org_w)
		self._name_to_top_map['mix_data'] = idx
		idx += 1
		
		top[idx].reshape(1)
		self._name_to_top_map['labels'] = idx
		idx += 1

		top[idx].reshape(1)
		self._name_to_top_map['clip_id'] = idx
		idx += 1

		print 'ModDataLayer: name_to_top:', self._name_to_top_map
		assert len(top) == len(self._name_to_top_map)		

	def forward(self, bottom, top):

		blobs = self._get_next_batch()

		for blob_name, blob in blobs.iteritems():
			top_ind = self._name_to_top_map[blob_name]
			# Reshape net's input blobs
			top[top_ind].reshape(*(blob.shape))
			# Copy data into net's input blobs
			top[top_ind].data[...] = blob.astype(np.float32, copy=False)

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass
