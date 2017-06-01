# Make sure that caffe is on the python path:
caffe_root = '/home/ls/mscnn/install/'  # this file is expected to be in {caffe_root}/examples
rcnn_lib = '/home/ls/mscnn/lib/'

import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, rcnn_lib)

import caffe
	
# MSCNN Load the original network and extract the fully connected layers' parameters.
mscnn_net = caffe.Net('/home/ls/mscnn/net_surgery/end2end/mscnn_deploy.prototxt','/home/ls/mscnn/net_surgery/mscnn_deploy.caffemodel', caffe.TRAIN)
mscnn_params_s = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3', 'conv6_1','loss1-conv1','LFCN_1_3x5', 'LFCN_1_5x7', 'LFCN_2_3x5', 'LFCN_2_5x7', 'LFCN_3_3x5', 'LFCN_3_5x7' ,'LFCN_4_3x5', 'fc6','cls_pred', 'bbox_pred']
mscnn_params_m = ['conv1_1']

# MSCNN fc_params = {name: (weights, biases)}
fc_mscnn_params_s = {pr: (mscnn_net.params[pr][0].data, mscnn_net.params[pr][1].data) for pr in mscnn_params_s}
fc_mscnn_params_m = {pr: (mscnn_net.params[pr][0].data, mscnn_net.params[pr][1].data) for pr in mscnn_params_m}

# Lenet
lenet_net = caffe.Net('/home/ls/mscnn/net_surgery/lenet/origin.prototxt','/home/ls/mscnn/net_surgery/lenet/lenet_iter_10000.caffemodel', caffe.TRAIN)
lenet_params_s = ['conv2'] #,'ip1', 'ip2']
lenet_params_m = ['conv1']

# MSCNN fc_params = {name: (weights, biases)}
fc_lenet_params_s = {pr: (lenet_net.params[pr][0].data, lenet_net.params[pr][1].data) for pr in lenet_params_s}
fc_lenet_params_m = {pr: (lenet_net.params[pr][0].data, lenet_net.params[pr][1].data) for pr in lenet_params_m}

# Load New net
new_net = caffe.Net('/home/ls/mscnn/net_surgery/end2end/mscnn_lenet/mscnn_lenet.prototxt', caffe.TRAIN)
new_params_s_mscnn = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3', 'conv6_1','loss1-conv1','LFCN_1_3x5', 'LFCN_1_5x7', 'LFCN_2_3x5', 'LFCN_2_5x7', 'LFCN_3_3x5', 'LFCN_3_5x7' ,'LFCN_4_3x5', 'fc6','cls_pred', 'bbox_pred'] 
new_params_m_mscnn = ['conv1_1_tem']

new_params_s_lenet = ['conv9'] #, 'ip1_mod', 'ip2_mod']
new_params_m_lenet = ['conv8']

# fc_params = {name: (weights, biases)}
new_fc_params_s_mscnn = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params_s_mscnn}
new_fc_params_m_mscnn = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params_m_mscnn}
new_fc_params_s_lenet = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params_s_lenet}
new_fc_params_m_lenet = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params_m_lenet}

# same part for mscnn part
for pr, new_pr in zip(mscnn_params_s, new_params_s_mscnn):
	new_fc_params_s_mscnn[new_pr][0][...]=fc_mscnn_params_s[pr][0] #weights
	new_fc_params_s_mscnn[new_pr][1][...]=fc_mscnn_params_s[pr][1] #bias
print "MSCNN same layers are done"

# same part for mscnn part
for pr, new_pr in zip(lenet_params_s, new_params_s_lenet):
	new_fc_params_s_lenet[new_pr][0][...]=fc_lenet_params_s[pr][0] #weights
	new_fc_params_s_lenet[new_pr][1][...]=fc_lenet_params_s[pr][1] #bias
print "LENET same layers are done"

# modify part
# give conv1_1_tem avergy conv1_1
for pr, new_pr in zip(mscnn_params_m, new_params_m_mscnn):
	for n in range (0, fc_mscnn_params_m[pr][0].shape[0]):
		for c_i in range (0, fc_mscnn_params_m[pr][0].shape[1]):
			new_fc_params_m_mscnn[new_pr][0][n][0] += fc_mscnn_params_m[pr][0][n][c_i]
		new_fc_params_m_mscnn[new_pr][0][n][0] /= 3
	new_fc_params_m_mscnn[new_pr][1][...]=fc_mscnn_params_m[pr][1]
print "Adaptive part of MSCNN is done"

# 
for pr, new_pr in zip(lenet_params_m, new_params_m_lenet):
	for n in range (0, fc_lenet_params_m[pr][0].shape[0]):
		for c_i in range (0, fc_lenet_params_m[pr][0].shape[1]):
			new_fc_params_m_lenet[new_pr][0][n][c_i] = fc_lenet_params_m[pr][0][n][c_i]
		new_fc_params_m_lenet[new_pr][0][n][1] = fc_lenet_params_m[pr][0][n][0]
	new_fc_params_m_lenet[new_pr][1][...]=fc_lenet_params_m[pr][1]

print "Adaptive part of MSCNN is done"

new_net.save('/home/ls/mscnn/net_surgery/end2end/mscnn_lenet/mscnn_lenet.caffemodel')

print 'channel convert is done'
