import numpy as np
import math
from nms.gpu_nms import gpu_nms
from ucfarg_cfg import ucfarg_cfg


def nms(dets, thresh):
	if dets.shape[0] == 0:
		return []
	new_dets = np.copy(dets)
	new_dets[:,2] += new_dets[:,0]
	new_dets[:,3] += new_dets[:,1]
	return gpu_nms(new_dets, thresh, device_id=ucfarg_cfg.GPU_ID)


def box_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def center_distance(boxA, boxB):
	x_A = (boxA[0] + boxA[2])/2
	y_A = (boxA[1] + boxA[3])/2
	
	x_B = (boxB[0] + boxB[2])/2
	y_B = (boxB[1] + boxB[3])/2

	return math.sqrt((x_B-x_A)**2 + (y_B-y_A)**2)


def comp_bbox(current_bbox, last_bbox):
	if len(current_bbox.shape) != 2 or len(last_bbox.shape) != 2 or current_bbox.shape[1] != 4 or last_bbox.shape[1] != 4:
		print "bbox mismatch by Ls in bbox_cal.py"
		raise
	skip_list = []
	for i in xrange(current_bbox.shape[0]):
		for j in xrange(last_bbox.shape[0]):
			if j in skip_list:
				break
			else:
				iou = box_iou(current_bbox[i], last_bbox[j])
				cd = center_distance(current_bbox[i], last_bbox[j])
				if iou > 0.45 and cd < 230:
					current_bbox[i][0] = min(current_bbox[i][0], last_bbox[j][0])
					current_bbox[i][1] = min(current_bbox[i][1], last_bbox[j][1])
					current_bbox[i][2] = max(current_bbox[i][2], last_bbox[j][2])
					current_bbox[i][3] = max(current_bbox[i][3], last_bbox[j][3])
					skip_list.append(j)
					break
	return current_bbox

def bbox_denormalize(bbox_pred, proposals, ratios, orgW, orgH):

	ctr_x = proposals[:,0]+0.5*proposals[:,2]
	ctr_y = proposals[:,1]+0.5*proposals[:,3]

	tx = bbox_pred[:,0] *proposals[:,2] + ctr_x
	ty = bbox_pred[:,1] *proposals[:,3] + ctr_y

	tw = proposals[:,2] * np.exp(bbox_pred[:,2])
	th = proposals[:,3] * np.exp(bbox_pred[:,3])

	#Fix Bug 2
	tx -= tw/2 
	ty -= th/2
	tx /= ratios[0] 
	tw /= ratios[0]
	ty /= ratios[1] 
	th /= ratios[1]

	tx[tx < 0] = 0
	ty[ty < 0] = 0
	#Fix Bug 3
	tw[tw > (orgW - tx)] = (orgW - tx[tw > (orgW - tx)])
	th[th > (orgH - ty)] = (orgH - ty[th > (orgH - ty)])
	new_boxes = np.hstack((tx[:, None], ty[:, None], tw[:, None], th[:, None])).astype(np.float32).reshape((-1, 4)) #suspecious
	return new_boxes

def filter_proposals(proposals, threshold=-10):
	#Bug 1 Fixed
	keeps = (proposals[:, -1] >= threshold) & (proposals[:, 2] != 0) & (proposals[:, 3] != 0)
	return keeps

def get_confidence(cls_pred):
	exp_score = np.exp(cls_pred)
	sum_exp_score = np.sum(exp_score, 1)
	confidence = exp_score[:, 1] / sum_exp_score

	return confidence
