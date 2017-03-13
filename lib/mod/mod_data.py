import numpy as np
import os
import cv2
from ucfarg_cfg import ucfarg_cfg

import caffe


class ModDataLayer(caffe.Layer):
	# """docstring for ModDataLayer"""


	def _get_next_batch(self):
	# """Return the blobs to be used for the next minibatch.

	# If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
	# separate process and made available through self._blob_queue.
	# """

		blob_spatial_im = np.zeros((ucfarg_cfg.TRAIN.IMS_PER_BATCH, ucfarg_cfg.TRAIN.SPATIAL_CHANNELS, ucfarg_cfg.TRAIN.TARGET_H, ucfarg_cfg.TRAIN.TARGET_W), dtype=np.float32)
		blob_mix_im = np.zeros((ucfarg_cfg.TRAIN.IMS_PER_BATCH, ucfarg_cfg.TRAIN.MIX_CHANNELS, ucfarg_cfg.TRAIN.ORG_H, ucfarg_cfg.TRAIN.ORG_W), dtype=np.float32)
		label_blob = np.zeros((0), dtype=np.float32)
		init_tag_blob = np.zeros((0), dtype=np.float32)

		for batch_index in xrange(ucfarg_cfg.TRAIN.IMS_PER_BATCH):

			if self._cur == len(self._indexlist):
				self._cur = 0

			index = self._indexlist[self._cur]
			mix_im = np.asarray(np.load(os.path.join(ucfarg_cfg.TRAIN.DATA_ROOT, index + ucfarg_cfg.TRAIN.DATA_EXTENSION)))

			spatial_im = np.asarray(mix_im[:, :, 0:-1])
			spatial_im = cv2.resize(spatial_im, (ucfarg_cfg.TRAIN.TARGET_W, ucfarg_cfg.TRAIN.TARGET_H)).astype(np.float32)
			spatial_im -= ucfarg_cfg.TRAIN.MEAN_3
			spatial_im = spatial_im.transpose((2,0,1))
			blob_spatial_im[batch_index, :, :, :] = spatial_im

			mix_im = mix_im.astype(np.float32)
			mix_im -= ucfarg_cfg.TRAIN.MEAN_4
			mix_im = mix_im.transpose((2,0,1))
			blob_mix_im[batch_index, :, :, :] = mix_im

			if mix_im.shape[0] != 4 or spatial_im.shape[0] !=3:
				print "image shape mismatch by Ls"
				raise

			label_file = os.path.join(ucfarg_cfg.TRAIN.LABEL_ROOT, index + ucfarg_cfg.TRAIN.LABEL_EXTENSION)
			assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)

			with open(label_file) as f:
				label_data = f.readline()

			label = int(label_data.split()[0])
			init_tag = int(label_data.split()[1])

			label_blob = np.hstack((label_blob, label))
			init_tag_blob = np.hstack((init_tag_blob, init_tag))

			self._cur += 1
		
		blobs = {'init_tag': init_tag_blob, 'labels': label_blob, 'mix_data': blob_mix_im, 'spatial_data': blob_spatial_im}
		return blobs

	def setup(self, bottom, top):
		# Setup ModDataLayer
		self._cur = 0
		self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TRAIN.LIST_FILE)]
		self._name_to_top_map = {}

		idx = 0
		top[idx].reshape(ucfarg_cfg.TRAIN.IMS_PER_BATCH, ucfarg_cfg.TRAIN.SPATIAL_CHANNELS,
			ucfarg_cfg.TRAIN.TARGET_H, ucfarg_cfg.TRAIN.TARGET_W)
		self._name_to_top_map['spatial_data'] = idx
		idx += 1

		top[idx].reshape(ucfarg_cfg.TRAIN.IMS_PER_BATCH, ucfarg_cfg.TRAIN.MIX_CHANNELS,
			ucfarg_cfg.TRAIN.ORG_H, ucfarg_cfg.TRAIN.ORG_W)
		self._name_to_top_map['mix_data'] = idx
		idx += 1
		
		top[idx].reshape(1)
		self._name_to_top_map['labels'] = idx
		idx += 1

		top[idx].reshape(1)
		self._name_to_top_map['init_tag'] = idx
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
