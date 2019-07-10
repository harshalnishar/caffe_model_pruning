'''
Caffe net-surgery for channel pruning based on L1 norm for simple layers without skip connections
List of layers needs to be specifed mannually

Author: Harshal
Created on : 10 July 2019
Modified on: 10 July 2019
'''

import caffe
import numpy as np
import cv2
from caffe.proto import caffe_pb2

# input test image path
image_path = '<image_path>' 
deploy_prototxt = 'deploy.prototxt' 
deploy_prototxt_modified = 'deploy_pruned.prototxt' 
caffemodel = '_iter_500000.caffemodel' 

test_net = caffe.Net(deploy_prototxt, caffemodel, caffe.TEST)
test_net_pruned = caffe.Net(deploy_prototxt_modified, caffe.TEST)

layer_names = list(test_net._layer_names)
layers_to_prune = ['res2a_branch2a']
channel_to_prune = None
for (key, value) in test_net.params.items():
# for key in ['conv1']:
    layer_type = test_net.layers[layer_names.index(key)].type
    print(layer_type)
    if layer_type == 'Convolution' and key in layers_to_prune:
        W = test_net.params[key][0].data
        print(W.shape)
        L1_norm = np.sum(np.abs(W), axis = (1, 2, 3))
        channel_to_prune = np.argmin(L1_norm)
        print('Pruning channel no. {}, norm {}'.format(channel_to_prune, L1_norm[channel_to_prune]))

        temp = np.zeros(W.shape, np.float32)
        np.copyto(temp, W)
        temp = np.delete(temp, channel_to_prune, axis = 0)
        # temp[channel_to_prune, :, :, :] = 0.0
        np.copyto(test_net_pruned.params[key][0].data, temp)

        if len(list(test_net.params[key])) == 2:   # check if bias paramter is also there
            b = test_net.params[key][1].data
            print(b.shape)
            temp = np.zeros(b.shape, np.float32)
            np.copyto(temp, b)
            temp = np.delete(temp, channel_to_prune, axis = 0)
            # temp[channel_to_prune, :, :, :] = 0.0
            np.copyto(test_net_pruned.params[key][1].data, temp)
    
    else:
        for i in range(len(list(test_net.params[key]))):
            print(test_net.params[key][i].data.shape)
            if channel_to_prune is not None and (i < 2 or layer_type == 'Scale'):
                temp = np.zeros(test_net.params[key][i].data.shape, np.float32)
                np.copyto(temp, test_net.params[key][i].data)
                if layer_type == 'Convolution' and i == 0:
                    temp = np.delete(temp, channel_to_prune, axis = 1)
                    channel_to_prune = None
                else:
                    temp = np.delete(temp, channel_to_prune, axis = 0)
            else:
                np.copyto(test_net_pruned.params[key][i].data, test_net.params[key][i].data)


test_net_pruned.save('_iter_500000_pruned.caffemodel')

# print(test_net.params['conv1'][0].data.shape)
# print(test_net.params['conv1'][1].data.shape)


#--------------------------------------------------------------------------------------------------------------#
data_image_1 = cv2.imread(image_path)

# data_image_1[:, :, 0] = data_image_1[:, : , 0].astype('float') - 193.24
# data_image_1[:, :, 1] = data_image_1[:, : , 1].astype('float') - 107.298
# data_image_1[:, :, 2] = data_image_1[:, : , 2].astype('float') - 162.016

# data_image_1[:, :, 0] = data_image_1[:, :, 2]
# data_image_1[:, :, 1] = data_image_1[:, :, 1]
# data_image_1[:, :, 2] = data_image_1[:, :, 0]

data_image = np.transpose(data_image_1, (2, 0, 1))
data_image = data_image.reshape((1, 3, 512, 512))

test_net.blobs['data'].data[...] = data_image
test_net.forward()

# out_image = test_net.blobs['score'].data
# out_image = test_net.blobs['deconvFusion_Sem_3'].data
out_image = test_net.blobs['argmax'].data
print(out_image)
# out_image = out_image.reshape((3, 512, 512))
out_image = out_image.reshape((512, 512)).astype('uint8')

# out_image = out_image.transpose((1, 2, 0))
out_image[out_image == 1.0] = 255

# out_image = (out_image - minimum) * 255 / (maximum - minimum)
print(out_image)
print(data_image.shape)

cv2.imwrite('test_out.png', out_image)

test_net_pruned.blobs['data'].data[...] = data_image
test_net_pruned.forward()

# out_image = test_net.blobs['score'].data
# out_image = test_net.blobs['deconvFusion_Sem_3'].data
out_image = test_net_pruned.blobs['argmax'].data
print(out_image)
# out_image = out_image.reshape((3, 512, 512))
out_image = out_image.reshape((512, 512)).astype('uint8')

# out_image = out_image.transpose((1, 2, 0))
out_image[out_image == 1.0] = 255

# out_image = (out_image - minimum) * 255 / (maximum - minimum)
print(out_image)
print(data_image.shape)

cv2.imwrite('test_out_pruned.png', out_image)
