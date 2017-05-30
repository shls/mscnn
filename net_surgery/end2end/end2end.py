# Make sure that caffe is on the python path:
caffe_root = '/home/ls/mscnn/install/'  # this file is expected to be in {caffe_root}/examples
rcnn_lib = '/home/ls/mscnn/lib/'

import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, rcnn_lib)

import caffe
	
# Load the original network and extract the fully connected layers' parameters.
mscnn_net = caffe.Net('/home/ls/mscnn/net_surgery/end2end/mscnn_deploy.prototxt','/home/ls/mscnn/net_surgery/mscnn_deploy.caffemodel', caffe.TRAIN)
mscnn_params_s = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3', 'conv6_1','loss1-conv1','LFCN_1_3x5', 'LFCN_1_5x7', 'LFCN_2_3x5', 'LFCN_2_5x7', 'LFCN_3_3x5', 'LFCN_3_5x7' ,'LFCN_4_3x5', 'fc6','cls_pred', 'bbox_pred']
mscnn_params_m = ['conv1_1']

# fc_params = {name: (weights, biases)}
fc_mscnn_params_s = {pr: (mscnn_net.params[pr][0].data, mscnn_net.params[pr][1].data) for pr in mscnn_params_s}
fc_mscnn_params_m = {pr: (mscnn_net.params[pr][0].data, mscnn_net.params[pr][1].data) for pr in mscnn_params_m}


# Load New net
new_net = caffe.Net('/home/ls/mscnn/net_surgery/end2end/train_mod_st2.prototxt', caffe.TRAIN)
new_mscnn_params_s = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3', 'conv6_1','loss1-conv1','LFCN_1_3x5', 'LFCN_1_5x7', 'LFCN_2_3x5', 'LFCN_2_5x7', 'LFCN_3_3x5', 'LFCN_3_5x7' ,'LFCN_4_3x5', 'fc6','cls_pred', 'bbox_pred'] 
new_mscnn_params_m = ['conv1_1_tem']


# fc_params = {name: (weights, biases)}
new_fc_mscnn_params_s = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_mscnn_params_s}
new_fc_mscnn_params_m = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_mscnn_params_m}

# same part for mscnn part
for pr, new_pr in zip(mscnn_params_s, new_mscnn_params_s):
	new_fc_mscnn_params_s[new_pr][0][...]=fc_mscnn_params_s[pr][0] #weights
	new_fc_mscnn_params_s[new_pr][1][...]=fc_mscnn_params_s[pr][1] #bias
print "All same layers are done"

# modify part
# give conv1_1_tem avergy conv1_1
for pr, new_pr in zip(mscnn_params_m, new_mscnn_params_m):
	for n in range (0, fc_mscnn_params_m[pr][0].shape[0]):
		for c_i in range (0, fc_mscnn_params_m[pr][0].shape[1]):
			new_fc_mscnn_params_m[new_pr][0][n][0] += fc_mscnn_params_m[pr][0][n][c_i]
		new_fc_mscnn_params_m[new_pr][0][n][0] /= 3
	new_fc_mscnn_params_m[new_pr][1][...]=fc_mscnn_params_m[pr][1]
print "The first layer of alexnet is done"

new_net.save('/home/ls/mscnn/net_surgery/end2end/mod_end2end.caffemodel')

print 'channel convert is done'
