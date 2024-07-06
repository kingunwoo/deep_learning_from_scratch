import numpy as np
import make_layer
from collections import OrderedDict

class LeNet_v1:
    def __init__(self,input,target):
        self.input = input
        self.target = target
        self.layers_dict = OrderedDict()
        #MNIST:input_shape(N,1,28,28)
        self.layers_dict['convolution_1'] = make_layer.convolution(C= 1, filter_n = 6, filter_h = 5,filter_w = 5, pad = 2, activation = 'sigmoid')
        #conv1_weight(6,1,5,5), conv1_bias(6,1,1), conv1_output(N,6,28,28)
        self.layers_dict['sigmoid1'] = make_layer.sigmoid()
        self.layers_dict['pooling_1'] = make_layer.pooling(pool_h = 2, pool_w = 2, mode = 'average', stride = 2)
        #pool1_output(N,14,14,6)
        
        self.layers_dict['convolution_2'] = make_layer.convolution(C = 6, filter_n = 16, filter_h = 5,filter_w = 5, activation = 'sigmoid')
        #conv2_weight(16,1,5,5), conv2_bias(16,1,1), conv2_output(N,16,10,10)
        self.layers_dict['sigmoid2'] = make_layer.sigmoid()
        self.layers_dict['pooling_2'] = make_layer.pooling(pool_h = 2, pool_w = 2,mode = 'average', stride = 2)
        #pool2_output(5,5,16)
        
        self.layers_dict['flatten1'] = make_layer.flatten()
        self.layers_dict['fc1'] = make_layer.fc(input_node= 400, output_node = 120)
        self.layers_dict['sigmoid3'] = make_layer.sigmoid()
        self.layers_dict['fc2'] = make_layer.fc(input_node= 120, output_node = 84)
        self.layers_dict['sigmoid4'] = make_layer.sigmoid()
        self.layers_dict['fc3'] = make_layer.fc(input_node=84, output_node =10)