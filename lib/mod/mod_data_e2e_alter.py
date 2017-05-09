import numpy as np
import os
import cv2
from ucfarg_cfg import ucfarg_cfg
import time
import caffe, json
import threading, Queue

class ModDataLayerE2EAlter(caffe.Layer):
	# """docstring for ModDataLayer"""


	def _load_data(self, queue, index):
		roi_pool_conv4_3 = np.asarray(np.load(os.path.join(self._spatial_prefeature_root, index + self._data_extension)))
		roi_pool_temporal_raw = np.asarray(np.load(os.path.join(self._temporal_prefeature_root, index + self._data_extension)))
		label_reshape = np.asarray(np.load(os.path.join(self._label_prefeature_root, index + self._data_extension)))
		queue.put((roi_pool_conv4_3, roi_pool_temporal_raw, label_reshape))

	def _get_next_batch(self):
	# """Return the blobs to be used for the next minibatch.

	# If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
	# separate process and made available through self._blob_queue.
	# """
		blob_spatial_fm = np.zeros((0, self._spatial_prefeature_channels,self._spatial_prefeature_height,self._spatial_prefeature_width), dtype=np.float32)
		blob_temporal_fm = np.zeros((0, self._temporal_prefeature_channels,self._temporal_prefeature_height, self._temporal_prefeature_width), dtype=np.float32)
		blob_label = np.zeros((0,1,1,1), dtype=np.float32)
		blob_index = np.zeros((0), dtype=np.float32)

		thread_list = []
		queue = Queue.Queue()

		for batch_index in xrange(self._imgs_per_batch):

			if self._cur == len(self._indexlist):
				self._cur = 0

			index = self._indexlist[self._cur]
			# roi_pool_conv4_3 = np.asarray(np.load(os.path.join(self._spatial_prefeature_root, index + self._data_extension)))
			# roi_pool_temporal_raw = np.asarray(np.load(os.path.join(self._temporal_prefeature_root, index + self._data_extension)))
			# label_reshape = np.asarray(np.load(os.path.join(self._label_prefeature_root, index + self._data_extension)))
			data_thread = threading.Thread(target=self._load_data, args=(queue,index))
			thread_list.append(data_thread)
			self._cur += 1

		for thread in thread_list:
			thread.start()
		
		start = time.time()
		for thread in thread_list:
			thread.join()
			roi_pool_conv4_3, roi_pool_temporal_raw, label_reshape = queue.get()

			blob_spatial_fm = np.append(blob_spatial_fm,roi_pool_conv4_3, axis=0)
#np.concatenate((blob_spatial_fm,roi_pool_conv4_3), axis=0)
			blob_temporal_fm = np.append(blob_temporal_fm,roi_pool_temporal_raw, axis=0)
#np.concatenate((blob_temporal_fm,roi_pool_temporal_raw), axis=0)
			blob_label = np.append(blob_label,label_reshape,axis=0)
#np.concatenate((blob_label,label_reshape),axis=0)
			blob_index = np.append(blob_index,np.full((len(label_reshape)), batch_index),axis=0)
#np.concatenate((blob_index,np.full((len(label_reshape)), batch_index)))
		end = time.time()
		print "Time eclapsed for data loading with thread: ", end - start, " s"
		
		print "blob_spatial_fm.shape", blob_spatial_fm.shape, "blob_temporal_fm.shape", blob_temporal_fm.shape, "blob_label.shape", blob_label.shape, "blob_index.shape", blob_index.shape
		blobs = {'roi_pool_spatial_con4_3': blob_spatial_fm, 'roi_pool_temporal_raw': blob_temporal_fm, 'label_reshape': blob_label, 'batch_index': blob_index}
		return blobs

	def setup(self, bottom, top):

		self._cur = 0
		self._name_to_top_map = {}

		if (self.phase == caffe.TRAIN):
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TRAIN.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TRAIN.E2E_ALTER_IMS_PER_BATCH
			self._data_extension = ucfarg_cfg.TRAIN.DATA_EXTENSION
			self._spatial_prefeature_root = ucfarg_cfg.TRAIN.SPATIAL_PREFEATURE_ROOT
			self._spatial_prefeature_width = ucfarg_cfg.TRAIN.SPATIAL_PREFEATURE_WIDTH
			self._spatial_prefeature_height = ucfarg_cfg.TRAIN.SPATIAL_PREFEATURE_HEIGHT
			self._spatial_prefeature_channels = ucfarg_cfg.TRAIN.SPATIAL_PREFEATURE_CHANNELS
			self._temporal_prefeature_root = ucfarg_cfg.TRAIN.TEMPORAL_PREFEATURE_ROOT
			self._temporal_prefeature_width = ucfarg_cfg.TRAIN.TEMPORAL_PREFEATURE_WIDTH
			self._temporal_prefeature_height = ucfarg_cfg.TRAIN.TEMPORAL_PREFEATURE_HEIGHT
			self._temporal_prefeature_channels = ucfarg_cfg.TRAIN.TEMPORAL_PREFEATURE_CHANNELS
			self._label_prefeature_root = ucfarg_cfg.TRAIN.LABEL_PREFEATURE_ROOT

		else:
			# Setup ModDataLayer
			self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TEST.LIST_FILE)]
			self._imgs_per_batch = ucfarg_cfg.TEST.E2E_ALTER_IMS_PER_BATCH
			self._data_extension = ucfarg_cfg.TEST.DATA_EXTENSION
			self._spatial_prefeature_root = ucfarg_cfg.TEST.SPATIAL_PREFEATURE_ROOT
			self._spatial_prefeature_width = ucfarg_cfg.TEST.SPATIAL_PREFEATURE_WIDTH
			self._spatial_prefeature_height = ucfarg_cfg.TEST.SPATIAL_PREFEATURE_HEIGHT
			self._spatial_prefeature_channels = ucfarg_cfg.TEST.SPATIAL_PREFEATURE_CHANNELS
			self._temporal_prefeature_root = ucfarg_cfg.TEST.TEMPORAL_PREFEATURE_ROOT
			self._temporal_prefeature_width = ucfarg_cfg.TEST.TEMPORAL_PREFEATURE_WIDTH
			self._temporal_prefeature_height = ucfarg_cfg.TEST.TEMPORAL_PREFEATURE_HEIGHT
			self._temporal_prefeature_channels = ucfarg_cfg.TEST.TEMPORAL_PREFEATURE_CHANNELS
			self._label_prefeature_root = ucfarg_cfg.TEST.LABEL_PREFEATURE_ROOT

		idx = 0
		top[idx].reshape(self._imgs_per_batch, self._spatial_prefeature_channels,
			self._spatial_prefeature_height,self._spatial_prefeature_width)
		self._name_to_top_map['roi_pool_spatial_con4_3'] = idx
		idx += 1

		top[idx].reshape(self._imgs_per_batch, self._temporal_prefeature_channels,
			self._temporal_prefeature_height, self._temporal_prefeature_width)
		self._name_to_top_map['roi_pool_temporal_raw'] = idx
		idx += 1
		
		top[idx].reshape(self._imgs_per_batch,1,1,1)
		self._name_to_top_map['label_reshape'] = idx
		idx += 1

		top[idx].reshape(1)
		self._name_to_top_map['batch_index'] = idx
		idx += 1

		print 'ModDataLayer: name_to_top:', self._name_to_top_map
		assert len(top) == len(self._name_to_top_map)		

	def forward(self, bottom, top):

		start = time.time()
		blobs = self._get_next_batch()
		end = time.time()
		print "Time eclapsed for data loading: ", end - start, " s" 

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
