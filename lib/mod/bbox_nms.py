import caffe
import numpy as np
from bbox_cal import nms, box_iou, center_distance, comp_bbox, bbox_denormalize, filter_proposals, get_confidence
from ucfarg_cfg import ucfarg_cfg


class BboxNMSLayer(caffe.Layer):
	"""docstring for BboxNMS"""
	def setup(self, bottom, top):
		self.mhi_buf = []
		self.cur_buf_id = 0
		self.last_clip_id = 0

		if (self.phase == caffe.TRAIN):
			 self._ratios = ucfarg_cfg.TRAIN.RATIOS
			 self._org_w = ucfarg_cfg.TRAIN.ORG_W
			 self._org_h = ucfarg_cfg.TRAIN.ORG_H
			 self._nms_thresh = ucfarg_cfg.TRAIN.NMS_THRESH
			 self._nms_confi_thresh = ucfarg_cfg.TRAIN.NMS_CONFI_THRESH
		else:
			 self._ratios = ucfarg_cfg.TEST.RATIOS
			 self._org_w = ucfarg_cfg.TEST.ORG_W
			 self._org_h = ucfarg_cfg.TEST.ORG_H
			 self._nms_thresh = ucfarg_cfg.TEST.NMS_THRESH
			 self._nms_confi_thresh = ucfarg_cfg.TEST.NMS_CONFI_THRESH

		top[0].reshape(1,5)
		top[1].reshape(1)

	def forward(self, bottom, top):

		# batch_idxs batch index (image index in batch)
		# proposals boundingbox proposals from predictions
		# cls_preds class predictions 
		# clip_ids clip id in all clips
		# lables image lable
		bbox_preds = bottom[0].data
		batch_idxs = bottom[1].data.reshape((-1,6))[:,:1]
		proposals = bottom[1].data.reshape((-1,6))[:,1:]
		proposals[:,2] -= proposals[:,0]
		proposals[:,3] -= proposals[:,1]
		cls_preds = bottom[2].data#
		clip_ids = bottom[3].data
		labels = bottom[4].data

		# top blob
		blob_rois = np.zeros((1,5), dtype=np.float32)
		blob_label = np.zeros((1,1,1,1), dtype=np.int)
		
		# turn img index in batch to number of proposals per img
		uni_batch_idxs, num_of_rois = np.unique(batch_idxs, return_counts=True)

		if len(num_of_rois) != len(clip_ids):
			print "shape mismatch,raised by Ls"
  			raise

		for batch_idx in xrange(len(num_of_rois)):

			cur_clip_id = clip_ids[batch_idx]
			
			if batch_idx == 0:
				rois_first = 0
				rois_end = num_of_rois[0]
			else:
				rois_first = sum(num_of_rois[0:batch_idx])
				rois_end = sum(num_of_rois[0:batch_idx+1])

			# rois of one batch(image)
			batch_proposals = proposals[rois_first:rois_end,]
			batch_bbox_pred = bbox_preds[rois_first:rois_end,]
			batch_cls_pred = cls_preds[rois_first:rois_end,]

			# filter poor rois of one batch(image)
			batch_keeps = filter_proposals(batch_proposals)
			batch_proposals = batch_proposals[batch_keeps]
			batch_bbox_pred =  batch_bbox_pred[batch_keeps]
			batch_cls_pred = batch_cls_pred[batch_keeps]
			
			# get the prediction boundingbox (0:4 background 4:8 pedestrian)
			pedestrian_batch_box_pred = batch_bbox_pred[:,4:8]
			# Denormalize the bbox
			boxes = bbox_denormalize(pedestrian_batch_box_pred, batch_proposals, self._ratios, self._org_w, self._org_h)
			# Get confidence
			confidence = get_confidence(batch_cls_pred)

			dets = np.hstack((boxes, confidence[:, np.newaxis])).astype(np.float32)
			keep_nms = nms(dets, self._nms_thresh)
			dets_nms = dets[keep_nms, :]
			boxes_nms = dets_nms[:, :4]
			inds = np.where(dets_nms[:, -1] >= self._nms_confi_thresh)[0]
			boxes_nms = dets_nms[inds, :4]

			# Enlarge boundingbox 
			# boxes_nms[:,0] -= boxes_nms[:,2]
			# boxes_nms[:,2] *= 3
			# boxes_nms[:,1] -= boxes_nms[:,3]/4
			# boxes_nms[:,3] *= 1.5

			boxes_nms_xyxy = boxes_nms[:]
			boxes_cache = np.zeros((boxes_nms.shape))
			boxes_cache[:,2:] = boxes_nms[:, 0:2]
			boxes_nms_xyxy += boxes_cache

			if self.last_clip_id != cur_clip_id:
				self.cur_buf_id = 0
				self.mhi_buf = []
				self.last_clip_id = cur_clip_id
				if boxes_nms_xyxy.shape[0] != 0:
					self.mhi_buf.append(boxes_nms_xyxy)
					self.cur_buf_id +=1
				if len(blob_rois) == 1 and np.count_nonzero(blob_rois) == 0:
					blob_rois[0,:] = np.array([batch_idx,0,0,128,128])
					blob_label[0,0,0,0] = 0
				else:
					blob_rois = np.concatenate((blob_rois, np.array([[batch_idx,0,0,128,128],])))
					blob_label = np.concatenate((blob_label,np.array([[[[0]]]])))
			else:
				if boxes_nms_xyxy.shape[0] != 0: 
					if len(self.mhi_buf) != 30:
						self.mhi_buf.append(boxes_nms_xyxy)
						self.cur_buf_id += 1
						blob_rois = np.concatenate((blob_rois, np.array([[batch_idx,0,0,128,128],])))
						blob_label = np.concatenate((blob_label,np.array([[[[0]]]])))
					else:
						for i in reversed(xrange(self.cur_buf_id)):
							boxes_nms_xyxy = comp_bbox(boxes_nms_xyxy, self.mhi_buf[i])
						for i in range(29,self.cur_buf_id-1,-1):
							boxes_nms_xyxy = comp_bbox(boxes_nms_xyxy, self.mhi_buf[i])

						if self.cur_buf_id == 30:
							self.cur_buf_id = 0
							self.mhi_buf[self.cur_buf_id] = boxes_nms_xyxy[:]
							self.cur_buf_id += 1
						else:
							self.mhi_buf[self.cur_buf_id] = boxes_nms_xyxy[:]
							self.cur_buf_id += 1

						#Compensate for batch index
						boxes_nms_xyxy = np.insert(boxes_nms_xyxy,0,batch_idx,axis=1)
						blob_rois = np.concatenate((blob_rois, boxes_nms_xyxy))
						blob_label = np.concatenate((blob_label, np.full((len(boxes_nms_xyxy),1,1,1), labels[batch_idx], dtype=np.float32)))

				else:
					if len(self.mhi_buf) == 30:
						if self.cur_buf_id == 30:
							self.cur_buf_id = 0
							self.mhi_buf[self.cur_buf_id] = self.mhi_buf[29][:]
							self.cur_buf_id += 1
						else:
							self.mhi_buf[self.cur_buf_id] = self.mhi_buf[self.cur_buf_id - 1][:]
							self.cur_buf_id += 1
					elif len(self.mhi_buf) != 0:
						self.mhi_buf.append(self.mhi_buf[self.cur_buf_id - 1])
						self.cur_buf_id += 1
					else:
						pass
					blob_rois = np.concatenate((blob_rois, np.array([[batch_idx,0,0,128,128],])))
					blob_label = np.concatenate((blob_label,np.array([[[[0]]]])))

		top[0].reshape(*(blob_rois.shape))
		top[0].data[...] = blob_rois.astype(np.float32, copy=False)

		top[1].reshape(*(blob_label.shape))
		top[1].data[...] = blob_label.astype(np.float32,copy=False)

	def reshape(self, bottom, top):
		pass

	def backward(self, top, propagate_down, bottom):
		# print top[0]._diff
		pass

