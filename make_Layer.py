import numpy as np
import activation_functions
import common_functions

class convolution:
    def __init__(self, filter_n ,filter_h, filter_w, pad = 0, stride = 1, activation = 'sigmoid'):
        self.filter_n = filter_n
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.pad = pad
        self.stride = stride
        if(activation == 'sigmoid' or activation == 'tanh'):
            self.filter = np.random.randn(filter_n, filter_h, filter_w) / np.sqrt(1 / self.filter_n * self.filter_h * self.filter_w)
        else:
            self.filter = np.random.randn(filter_n, filter_h, filter_w) / np.sqrt(2 / self.filter_n * self.filter_h * self.filter_w)
        
    def forward(self,input_data):
        N, C, H, W = input_data.shape
        OH = (H + 2 * self.pad - self.filter_h) // self.stride + 1
        OW = (W + 2 * self.pad - self.filter_w) // self.stride + 1
        
        flatten_img = common_functions.im2col(input_data, self.filter_h, self.filter_w, self.stride, self.pad)
        print(flatten_img.shape)
        flatten_filter = self.filter.reshape(self.filter_n, -1).T
        print(flatten_filter.shape)
        flatten_conv_img = np.dot(flatten_img, flatten_filter)
        print(flatten_conv_img.shape)
        output_img = flatten_conv_img.reshape(N,self.filter_n,OH,OW)
        return output_img
        
    def backward(self):
        
        return 

#lass fc:
#   def __init__(self,input_data,length_output_node,activation): 
#       self.output = np.ones(length_output_node)
#       self.input = input_data.reshape(-1,784)
#       self.activation = activation
#       if(activation == activation_functions.sigmoid or activation == activation_functions.tanh):
#       else:
#           gfg
#   def forward(self):
#       output = np.dot(input,weight) + bias
#       return output        
#   def backward(self):
#       return
#   
#lass pooling:
#   def __init__(self, pool_h, pool_w,  mode='max', pad = 0, stride = 1):
    