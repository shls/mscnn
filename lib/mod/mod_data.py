import caffe
import numpy as np
from ucfarg_cfg import ucfarg_cfg



class ModDataLayer(caffe.Layer):
	# """docstring for ModDataLayer"""


	def _get_next_batch(self):
	# """Return the blobs to be used for the next minibatch.

	# If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
	# separate process and made available through self._blob_queue.
	# """
		if self._cur == len(self._indexlist):
			self._cur = 0

		index = self._indexlist[self._cur]
		mix_im = np.asarray(np.load(os.path.join(ucfarg_cfg.TRAIN.DATA_ROOT, index, ucfarg_cfg.TRAIN.DATA_EXTENSION)))

		spatial_im = mix_im[:, :, :2].copy
		spatial_im = cv2.resize(im, (ucfarg_cfg.TRAIN.TARGET_W, ucfarg_cfg.TRAIN.TARGET_H)).astype(np.float32)
		spatial_im -= ucfarg_cfg.TRAIN.MEAN
		spatial_im = spatial_im.transpose((2,0,1))

		mix_im = mix_im.transpose((2,0,1))

		if mix_im.shape != 4 or spatial_im.shape !=3:
			print "image shape mismatch by Ls"
			raise

		lable_file = os.path.join(ucfarg_cfg.TRAIN.LABLE_ROOT, index, ucfarg_cfg.TRAIN.LABLE_EXTENSION)
		assert os.path.exists(lable_file), 'Path does not exist: {}'.format(lable_file)

		with open(lable_file) as f:
			lable_data = f.readline()

		lable = int(lable_data.split()[0])
		init_tag = int(lable_data.split()[1])

		blobs = {'init_tag': init_tag, 'labels': lable, 'mix_data': mix_im, 'spatial_data': spatial_im}
		
		self._cur += 1
		return blobs

	def setup(self, bottom, top):
		# Setup ModDataLayer
		self._cur = 0
		self._indexlist = [line.rstrip('\n') for line in open(ucfarg_cfg.TRAIN.LIST_FILE)]
		self._name_to_top_map = {}

		idx = 0
		top[idx].reshape(ucfarg_cfg.TRAIN.IMS_PER_BATCH, ucfarg_cfg.TRAIN.SPATIAL_CHANNELS,
			ucfarg_cfg.TRAIN.WIDTH, ucfarg_cfg.TRAIN.HEIGHT)
		self._name_to_top_map['spatial_data'] = idx
		idx += 1

		top[idx].reshape(ucfarg_cfg.TRAIN.IMS_PER_BATCH, ucfarg_cfg.TRAIN.MIX_CHANNELS,
			ucfarg_cfg.TRAIN.WIDTH, ucfarg_cfg.TRAIN.HEIGHT)
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
			top[top_ind].reshape(blob.shape)
			# Copy data into net's input blobs
			top[top_ind].data[...] = blob.astype(np.float32, copy=False)

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass
