import caffe
import numpy as np
from bbox_cal import nms, box_iou, center_distance, comp_bbox, bbox_denormalize, filter_proposals, get_confidence
from ucfarg_cfg import ucfarg_cfg


class BboxNMSLayer(caffe.Layer):
	"""docstring for BboxNMS"""
	def setup(self, bottom, top):
		self._buf = []
		self._cur = 0
		self._last_event_index = 0
		top[0].reshape(1,5)
		top[1].reshape(1)

	def forward(self, bottom, top):

		bbox_preds = bottom[0].data
		proposals = bottom[1].data.reshape((-1,6))[:,1:]
		proposals[:,2] -= proposals[:,0]
		proposals[:,3] -= proposals[:,1]
		cls_preds = bottom[2].data
		num_rois = bottom[3].data
		event_indexs = bottom[4].data
		labels = bottom[5].data

		blob_rois = np.zeros((1,5), dtype=np.float32)
		blob_label = np.zeros((1,1,1,1), dtype=np.int)

		for batch_index in xrange(len(event_indexs)):

			event_index = event_indexs[batch_index]
			
			if batch_index == 0:
				batch_start = 0
				batch_end = num_rois[0]
			else:
				batch_start = sum(num_rois[0:batch_index])
				batch_end = sum(num_rois[0:batch_index+1])

			batch_proposals = proposals[batch_start:batch_end,]

			batch_keeps = filter_proposals(batch_proposals)
			batch_bbox_pred =  bbox_preds[batch_keeps]
			batch_cls_pred = cls_preds[batch_keeps]
			batch_proposals = batch_proposals[batch_keeps]

			batch_pedestrian_boxes = batch_bbox_pred[:,4:8]

			#Denormalize the bbox and confidence
			boxes = bbox_denormalize(batch_pedestrian_boxes, batch_proposals, ucfarg_cfg.TRAIN.RATIOS, ucfarg_cfg.TRAIN.ORG_W, ucfarg_cfg.TRAIN.ORG_H)
			confidence = get_confidence(batch_cls_pred)

			dets = np.hstack((boxes, confidence[:, np.newaxis])).astype(np.float32)
			keep_nms = nms(dets, ucfarg_cfg.TRAIN.NMS_THRESH)
			dets_nms = dets[keep_nms, :]
			boxes_nms = dets_nms[:, :4]
			inds = np.where(dets_nms[:, -1] >= ucfarg_cfg.TRAIN.NMS_CONFI_THRESH)[0]
			boxes_nms = dets_nms[inds, :4]

			boxes_nms[:,0] -= boxes_nms[:,2]
			boxes_nms[:,2] *= 3
			boxes_nms[:,1] -= boxes_nms[:,3]/4
			boxes_nms[:,3] *= 1.5

			boxes_nms_xyxy = boxes_nms[:]
			boxes_cache = np.zeros((boxes_nms.shape))
			boxes_cache[:,2:] = boxes_nms[:, 0:2]
			boxes_nms_xyxy += boxes_cache

			if self._last_event_index != event_index:
				self._cur = 0
				self._buf = []
				self._last_event_index = event_index
				if boxes_nms_xyxy.shape[0] != 0:
					self._buf.append(boxes_nms_xyxy)
					self._cur +=1
				if len(blob_rois) == 1 and np.count_nonzero(blob_rois) == 0:
					blob_rois[0,:] = np.array([batch_index,0,0,128,128])
					blob_label[0,0,0,0] = 0
				else:
					blob_rois = np.concatenate(blob_rois, np.array([[batch_index,0,0,128,128],]))
					blob_label = np.concatenate(blob_label,np.array([[[[0]]]]))
			else:
				if boxes_nms_xyxy.shape[0] != 0:
					if len(self._buf) != 30:
						self._buf.append(boxes_nms_xyxy)
						self._cur += 1
						blob_rois = np.concatenate(blob_rois, np.array([[batch_index,0,0,128,128],]))
						blob_label = np.concatenate(blob_label,np.array([[[[0]]]]))
					else:
						for i in reversed(xrange(self._cur)):
							boxes_nms_xyxy = comp_bbox(boxes_nms_xyxy, self._buf[i])
						for i in range(29,self._cur-1,-1):
							boxes_nms_xyxy = comp_bbox(boxes_nms_xyxy, self._buf[i])

						if self._cur == 30:
							self._cur = 0
							self._buf[self._cur] = boxes_nms_xyxy[:]
							self._cur += 1
						else:
							self._buf[self._cur] = boxes_nms_xyxy[:]
							self._cur += 1

						#Compensate for batch index
						boxes_nms_xyxy = np.insert(boxes_nms_xyxy,0,batch_index,axis=1)
						
						blob_rois = np.concatenate(blob_rois, boxes_nms_xyxy)
						blob_label = np.concatenate(blob_label, np.full((len(boxes_nms_xyxy),1,1,1), labels[batch_index]))

				else:
					if len(self._buf) == 30:
						if self._cur == 30:
							self._cur = 0
							self._buf[self._cur] = self._buf[29][:]
							self._cur += 1
						else:
							self._buf[self._cur] = self._buf[self._cur - 1][:]
							self._cur += 1
					elif len(self._buf) != 0:
						self._buf.append(self._buf[self._cur - 1])
						self._cur += 1
					
					blob_rois = np.concatenate(blob_rois, np.array([[batch_index,0,0,128,128],]))
					blob_label = np.concatenate(blob_label,np.array([[[[0]]]]))
	

		top[0].reshape(len(blob_rois),5)
		top[1].reshape(len(blob_label),1,1,1)

		top[0].data[...] = blob_rois[:]
		top[1].data[...] = blob_label[:]



		# batch_keeps = filter_proposals(proposals)
		# batch_bbox_pred =  batch_bbox_pred[batch_keeps]
		# batch_cls_pred = batch_cls_pred[batch_keeps]
		# proposals = proposals[batch_keeps]

		# batch_pedestrian_boxes = batch_bbox_pred[:,4:8]

		# #Denormalize the bbox and confidence
		# boxes = bbox_denormalize(batch_pedestrian_boxes, proposals, ucfarg_cfg.TRAIN.RATIOS, ucfarg_cfg.TRAIN.ORG_W, ucfarg_cfg.TRAIN.ORG_H)
		# confidence = get_confidence(batch_cls_pred)

		# dets = np.hstack((boxes, confidence[:, np.newaxis])).astype(np.float32)
		# keep_nms = nms(dets, ucfarg_cfg.TRAIN.NMS_THRESH)
		# dets_nms = dets[keep_nms, :]
		# boxes_nms = dets_nms[:, :4]
		# inds = np.where(dets_nms[:, -1] >= ucfarg_cfg.TRAIN.NMS_CONFI_THRESH)[0]
		# boxes_nms = dets_nms[inds, :4]

		# boxes_nms[:,0] -= boxes_nms[:,2]
		# boxes_nms[:,2] *= 3
		# boxes_nms[:,1] -= boxes_nms[:,3]/4
		# boxes_nms[:,3] *= 1.5

		# boxes_nms_xyxy = boxes_nms[:]
		# boxes_cache = np.zeros((boxes_nms.shape))
		# boxes_cache[:,2:] = boxes_nms[:, 0:2]
		# boxes_nms_xyxy += boxes_cache

		# if self._last_event_index != event_index:
		# 	self._cur = 0
		# 	self._buf = []
		# 	self._last_event_index = event_index
		# 	if boxes_nms_xyxy.shape[0] != 0:
		# 		self._buf.append(boxes_nms_xyxy)
		# 		self._cur +=1
		# 	top[0].data[...] = np.array([[0,0,0,128,128],])
		# 	top[1].data[...] = np.array([[[[0]]]])
		# else:
		# 	if boxes_nms_xyxy.shape[0] != 0:
		# 		if len(self._buf) != 30:
		# 			self._buf.append(boxes_nms_xyxy)
		# 			self._cur += 1
		# 			top[0].data[...] = np.array([[0,0,0,128,128],])
		# 			top[1].data[...] = np.array([[[[0]]]])
		# 		else:
		# 			for i in reversed(xrange(self._cur)):
		# 				boxes_nms_xyxy = comp_bbox(boxes_nms_xyxy, self._buf[i])
		# 			for i in range(29,self._cur-1,-1):
		# 				boxes_nms_xyxy = comp_bbox(boxes_nms_xyxy, self._buf[i])

		# 			if self._cur == 30:
		# 				self._cur = 0
		# 				self._buf[self._cur] = boxes_nms_xyxy[:]
		# 				self._cur += 1
		# 			else:
		# 				self._buf[self._cur] = boxes_nms_xyxy[:]
		# 				self._cur += 1

		# 			#Compensate for batch index
		# 			boxes_nms_xyxy = np.insert(boxes_nms_xyxy,0,0,axis=1)

		# 			top[0].reshape(len(boxes_nms_xyxy),5)
		# 			top[1].reshape(len(boxes_nms_xyxy),1,1,1)

		# 			top[0].data[...] = boxes_nms_xyxy[:]
		# 			top[1].data[...] = np.full((len(boxes_nms_xyxy),1,1,1), labels)

		# 	else:
		# 		if len(self._buf) == 30:
		# 			if self._cur == 30:
		# 				self._cur = 0
		# 				self._buf[self._cur] = self._buf[29][:]
		# 				self._cur += 1
		# 			else:
		# 				self._buf[self._cur] = self._buf[self._cur - 1][:]
		# 				self._cur += 1
		# 		elif len(self._buf) != 0:
		# 			self._buf.append(self._buf[self._cur - 1])
		# 			self._cur += 1
				
		# 		top[0].data[...] = np.array([[0,0,0,128,128],])
		# 		top[1].data[...] = np.array([[[[0]]]])

	def reshape(self, bottom, top):
		pass

	def backward(self, top, propagate_down, bottom):
		# print top[0]._diff
		pass

