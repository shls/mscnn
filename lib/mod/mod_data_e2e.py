import numpy as np
import os
import cv2
from ucfarg_cfg import ucfarg_cfg
import time
import caffe, json
import threading, Queue

class ModDataLayerE2E(caffe.Layer):
	# """docstring for ModDataLayer"""


	def _load_data(self, spatial_im, temporal_im, label, clip_id,  b_index, bbox_num_org, label_reshape, index, count, batch_id, filename):
		for i in xrange(count):

			mix = np.asarray(np.load(os.path.join(self._data_root, index + self._data_extension)))
			spatial = np.asarray(mix_im[:, :, 0:-1])
			spatial = cv2.resize(spatial, (self._target_w, self._target_h)).astype(np.float32)
			spatial -= self._mean_3
			spatial = spatial.transpose((2,0,1))

			temporal = np.asarray(mix_im[:, :, -1])
			temporal = cv2.resize(temporal, (self._target_w, self._target_h)).astype(np.float32)
			temporal = self._mean_1
			temporal = np.expand_dims(temporal_im, axis=0)

			label_file = os.path.join(self._label_root, index + self._label_extension)
			assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)

			with open(label_file) as f:
				label_data = f.readline()

			spatial_im[batch_id] = spatial
			temporal_fm[batch_id] = temporal
			label[batch_id] = int(label_data.split()[0])
			clip_id[batch_id] = int(label_data.split()[1])
			bbox_num_org[index+i] = np.asarray(np.load(os.path.join(self._bbox_num_org_root, filename + self._data_extension)))[i][:]
			bbox_num_org[index+i][0] = batch_id
			label_reshape[index+i] = np.asarray(np.load(os.path.join(self._label_prefeature_root, filename + self._data_extension)))[i][:]

	def _get_count(self, results, index, id):
		results[id] = len(np.asarray(np.load(os.path.join(self._label_prefeature_root, index + self._data_extension))))


	def _get_next_batch(self):
	# """Return the blobs to be used for the next minibatch.

	# If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
	# separate process and made available through self._blob_queue.
	# """
		# start = time.time()

		# count_batch_bbox: records the number of bboxes per image
		# batch_index_bbox: records the index in the batch per image

		count_batch_bbox = [None] * self._imgs_per_batch
		batch_index_bbox = [None] * self._imgs_per_batch
		count_thread_list = []
		cur_index = self._cur

		for i in xrange(self._imgs_per_batch):
			if cur_index == len(self._indexlist):
				cur_index = 0

			index = self._indexlist[cur_index]
			count_thread = threading.Thread(target=self._get_count, args=(count_batch_bbox, index, i))
			count_thread_list.append(count_thread)
			cur_index += 1

		for thread in count_thread_list:
			thread.start()

		for thread in count_thread_list:
			thread.join()

		total_batch = sum(count_batch_bbox)


		for i in xrange(self._imgs_per_batch):
			batch_index_bbox[i] = sum(count_batch_bbox[:i])

		# end = time.time()
		# print "Time eclapsed for the first part: ", end - start, " s"

		# start = time.time()

		blob_spatial_im = np.zeros((self._imgs_per_batch, self._spatial_channels, self._target_h,self._target_w), dtype=np.float32)
		blob_temporal_im = np.zeros((self._imgs_per_batch, self._temporal_channels, self._target_h,self._target_w), dtype=np.float32)
		blob_lables = np.zeros((self._imgs_per_batch), dtype=np.float32)
		blob_clip_id = np.zeros((self._imgs_per_batch), dtype=np.float32)
		blob_bbox_num_org = np.zeros((total_batch,6), dtype=np.float32)
		blob_label_reshape = np.zeros((total_batch), dtype=np.float32)

		thread_list = []

		for im_batch_id in xrange(self._imgs_per_batch):

			if self._cur == len(self._indexlist):
				self._cur = 0

			im_name = self._indexlist[self._cur]
			# Fetch data in parallel 
			data_thread = threading.Thread(target=self._load_data, args=(blob_spatial_im, blob_temporal_im, blob_lables, blob_clip_id, blob_bbox_num_org, blob_label_reshape, batch_index_bbox[im_batch_id], count_batch_bbox[im_batch_id], im_batch_id, im_name))
			thread_list.append(data_thread)
			self._cur += 1

		for thread in thread_list:
			thread.start()

		for thread in thread_list:
			thread.join()

		# end = time.time()
		# print "Time eclapsed for the second part: ", end - start, " s"

		# print "blob_spatial_fm.shape", blob_spatial_fm.shape, "blob_temporal_fm.shape", blob_temporal_fm.shape, "blob_label.shape", blob_label.shape, "blob_index.shape", blob_index.shape
		blobs = {'spatial_data': blob_spatial_im, 'temporal_data': blob_temporal_im, 'labels': blob_lables, 'clip_id': blob_clip_id, "bbox_num_org": blob_bbox_num_org, 'label_reshape': blob_label_reshape}
		return blobs

	def setup(self, bottom, top):

		self._cur = 0
		self._name_to_top_map = {}

		if (self.phase == caffe.TRAIN):
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TRAIN.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TRAIN.E2E_ALTER_IMS_PER_BATCH
			self._data_extension = ucfarg_cfg.TRAIN.DATA_EXTENSION
			self._label_prefeature_root = ucfarg_cfg.TRAIN.LABEL_PREFEATURE_ROOT
			self._spatial_channels = ucfarg_cfg.TRAIN.SPATIAL_CHANNELS
			self._temporal_channels = ucfarg_cfg.TRAIN.TEMPORAL_CHANNELS
			self._target_h = ucfarg_cfg.TRAIN.TARGET_H
			self._target_w = ucfarg_cfg.TRAIN.TARGET_W
			self._data_root = ucfarg_cfg.TRAIN.DATA_ROOT
			self._mean_3 = ucfarg_cfg.TRAIN.MEAN_3
			self._mean_1 = ucfarg_cfg.TRAIN.MEAN_1
			self._label_root = ucfarg_cfg.TRAIN.LABEL_ROOT
			self._label_extension = ucfarg_cfg.TRAIN.LABEL_EXTENSION
			self._bbox_num_org_root = ucfarg_cfg.TRAIN.BBOX_NUM_ORG_ROOT 

		else:
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TEST.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TEST.E2E_ALTER_IMS_PER_BATCH
			self._data_extension = ucfarg_cfg.TEST.DATA_EXTENSION
			self._label_prefeature_root = ucfarg_cfg.TEST.LABEL_PREFEATURE_ROOT
			self._spatial_channels = ucfarg_cfg.TEST.SPATIAL_CHANNELS
			self._temporal_channels = ucfarg_cfg.TEST.TEMPORAL_CHANNELS
			self._target_h = ucfarg_cfg.TEST.TARGET_H
			self._target_w = ucfarg_cfg.TEST.TARGET_W
			self._data_root = ucfarg_cfg.TEST.DATA_ROOT
			self._mean_3 = ucfarg_cfg.TEST.MEAN_3
			self._mean_1 = ucfarg_cfg.TEST.MEAN_1
			self._label_root = ucfarg_cfg.TEST.LABEL_ROOT
			self._label_extension = ucfarg_cfg.TEST.LABEL_EXTENSION
			self._bbox_num_org_root = ucfarg_cfg.TEST.BBOX_NUM_ORG_ROOT 

		idx = 0
		top[idx].reshape(self._imgs_per_batch, self._spatial_channels,
			self._target_h,self._target_w)
		self._name_to_top_map['spatial_data'] = idx
		idx += 1

		top[idx].reshape(self._imgs_per_batch, self._temporal_channels,
			self._org_h, self._org_w)
		self._name_to_top_map['temporal_data'] = idx
		idx += 1
		
		top[idx].reshape(1)
		self._name_to_top_map['labels'] = idx
		idx += 1

		top[idx].reshape(1)
		self._name_to_top_map['clip_id'] = idx
		idx += 1
		
		top[idx].reshape(self._imgs_per_batch,1,1,1)
		self._name_to_top_map['bbox_num_org'] = idx
		idx += 1

		top[idx].reshape(self._imgs_per_batch,1,1,1)
		self._name_to_top_map['label_reshape'] = idx
		idx += 1

		print 'ModDataLayer: name_to_top:', self._name_to_top_map
		assert len(top) == len(self._name_to_top_map)		

	def forward(self, bottom, top):

		# start = time.time()
		blobs = self._get_next_batch()
		# end = time.time()
		# print "Time eclapsed for data loading: ", end - start, " s" 

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
