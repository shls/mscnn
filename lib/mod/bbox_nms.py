import caffe
import numpy as np
from bbox_cal import nms, box_iou, center_distance, comp_bbox, bbox_denormalize, filter_proposals, get_confidence
from ucfarg_cfg import ucfarg_cfg


class BboxNMSLayer(caffe.Layer):
	"""docstring for BboxNMS"""
	def setup(self, bottom, top):
		self._buf = []
		self._cur = 0
		top[0].reshape(1,5)
		top[1].reshape(1)

	def forward(self, bottom, top):

		bbox_pred = bottom[0].data
		proposals = bottom[1].data.reshape((-1,6))[:,1:]
		proposals[:,2] -=   proposals[:,0]
		proposals[:,3] -=   proposals[:,1]
		cls_pred = bottom[2].data
		init_tag = bottom[3].data
		labels = bottom[4].data

		keeps = filter_proposals(proposals)
		bbox_pred =  bbox_pred[keeps]
		cls_pred = cls_pred[keeps]
		proposals = proposals[keeps]

		pedestrian_boxes = bbox_pred[:,4:8]

		#Denormalize the bbox and confidence
		boxes = bbox_denormalize(pedestrian_boxes, proposals, ucfarg_cfg.TRAIN.RATIOS, ucfarg_cfg.TRAIN.ORG_W, ucfarg_cfg.TRAIN.ORG_H)
		confidence = get_confidence(cls_pred)

		dets = np.hstack((boxes, confidence[:, np.newaxis])).astype(np.float32)
		keep_nms = nms(dets, ucfarg_cfg.TRAIN.NMS_THRESH)
		dets_nms = dets[keep_nms, :]
		boxes_nms = dets_nms[:, :4]
		inds = np.where(dets_nms[:, -1] >= ucfarg_cfg.TRAIN.NMS_CONFI_THRESH)[0]
		boxes_nms = dets_nms[inds, :4]

		if boxes_nms.shape[0] == 0:
			top[0].data[...] = np.array(([0,0,0,128,128]))
			top[1].data[...] = np.array([0])
		if init_tag:
			self._buf.append(boxes_nms)
			top[0].data[...] = np.array(([0,0,0,128,128]))
			top[1].data[...] = np.array([0])
		else:
			print len(self._buf)
			print self._cur
			for i in reversed(xrange(self._cur)):
				boxes_nms = comp_bbox(boxes_nms, self._buf[i])
			for i in range(29,self._cur-1,-1):
				boxes_nms = comp_bbox(boxes_nms, self._buf[i])

			#Compensate for batch index
			boxes_nms = np.insert(boxes_nms,0,0,axis=1)
			print boxes_nms.shape

			top[0].reshape(len(boxes_nms),5)
			top[1].reshape(len(boxes_nms))

			top[0].data[...] = boxes_nms
			top[1].data[...] = np.full((len(boxes_nms)), labels)

	def reshape(self, bottom, top):
		pass

	def backward(self, bottom, top):
		pass

