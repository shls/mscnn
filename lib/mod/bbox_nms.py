import caffe
import numpy as np
from bbox_cal import nms, box_iou, center_distance, comp_bbox, filter_proposals, get_confidence
from ucfarg_cfg import ucfarg_cfg


class BboxNMSLayer(caffe.Layer):
	"""docstring for BboxNMS"""
	def setup(self, bottom, up):
		self._buf = []
		self._cur = 0
		pass

	def forward(self, bottom, up):

		bbox_pred = bottom[0].data
		proposals = bottom[1].data.reshape((-1,6))[:,1:]
		proposals[:,2] -=   proposals[:,0]
		proposals[:,3] -=   proposals[:,1]
		cls_pred = bottom[2].data
		init_tag = bottom[3].data

		keeps = filter_proposals(proposals)
		bbox_pred =  bbox_pred[keeps]
		cls_pred = cls_pred[keeps]
		proposals = proposals[keeps]

		pedestrian_boxes = bbox_pred[:,4:8]

		#Denormalize the bbox and confidence
		boxes = bbox_denormalize(pedestrian_boxes, proposals, ucfarg_cfg.TRAIN.RATIOS, ucfarg_cfg.TRAIN.ORG_W, ucfarg_cfg.TRAIN.ORG_H)
		confidence = get_confidence(cls_pred)

		dets = np.hstack((boxes, confidence[:, np.newaxis])).astype(np.float32)
		keep_nms = nms(dets, 0.3)
		dets_nms = dets[keep_nms, :]
		boxes_nms = dets_nms[i, :4]

		if boxes_nms.shape[0] == 0:
			top[0].data[...] = np.array([])
		if init_tag:
			self._buf.append(boxes_nms)
			top[0].data[...] = np.array([])
		else:
			for i in reversed(xrange(self._cur)):
				boxes_nms = comp_bbox(boxes_nms, self._buf[i])
			for i in range(30,self._cur,-1):
				boxes_nms = comp_bbox(boxes_nms, self._buf[i])

			#Compensate for batch index
			boxes_nms = np.insert(boxes_nms,0,0,axis=1)

			top[0].data[...] = boxes_nms

	def reshape(self, bottom, up):
		pass

	def backward(self, bottom, up):
		pass
