import numpy as np
import common_functions

class convolution:
    def __init__(self, C, filter_n ,filter_h, filter_w, pad = 0, stride = 1, activation = 'sigmoid'):
        self.filter_n = filter_n
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.pad = pad
        self.stride = stride
        if(activation == 'sigmoid' or activation == 'tanh'):
            self.filter_weight = np.random.randn(filter_n, C, filter_h, filter_w) / np.sqrt(1 / (self.filter_n * C * self.filter_h * self.filter_w))
        else:
            self.filter_weight = np.random.randn(filter_n, C, filter_h, filter_w) / np.sqrt(2 / (self.filter_n * C * self.filter_h * self.filter_w))

        print(self.filter_weight.shape)    
        self.bias = 0
        
    def forward(self,input_data):
        
        N, C, H, W = input_data.shape
        print(input_data.shape)
        OH = (H + 2 * self.pad - self.filter_h) // self.stride + 1
        OW = (W + 2 * self.pad - self.filter_w) // self.stride + 1
        
        flatten_img = common_functions.im2col(input_data, self.filter_h, self.filter_w, self.stride, self.pad)
        
        flatten_filter = self.filter_weight.reshape(self.filter_n, -1).T
        #ï¿½Ù·ï¿½ ï¿½ï¿½ï¿½ï¿½ flatten_filterï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½Ï´ï¿? ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ß¸ï¿½ï¿½Ç¾ï¿½ï¿½ï¿½ 
        print(flatten_filter.shape)
        print(flatten_img.shape)
        flatten_conv_img = np.dot(flatten_img, flatten_filter) + self.bias
        output_img = flatten_conv_img.reshape(N,self.filter_n,OH,OW)
        return output_img
        
   #def backward(self):
   #    
   #    return 

class fc:
    def __init__(self, input_node, output_node, activation = 'sigmoid'): 
        self.output_node = output_node
        self.activation = activation
        self.input_node = input_node
        weight_length = self.input_node * output_node
        if(activation == 'sigmoid' or activation == 'tanh'):
            self.weight = np.random.randn(input_node, output_node) / np.sqrt(1 / weight_length)
        else:
            self.weight = np.random.randn(input_node, output_node) / np.sqrt(2 / weight_length)
        self.bias = np.ones((self.output_node))
        
        #when using forward function
        self.input_data = None
        
        #when using backward funcion
        self.dWeight = None
        self.dBias = None
        
    def forward(self, input_data):
        self.input_data = input_data
        output = np.dot(input_data, self.weight) + self.bias
        return output
            
    #def backward(self, dout):
    #    dx = np.dot(dout, self.dWeight.T)
    #    self.dWeight = np.dout(self.input_data.T, dout)
    #    self.dBias = np.sum(dout, axis = 0)        
    #    return dx
    
class pooling:
    def __init__(self, pool_h, pool_w,  mode='max', pad = 0, stride = 1):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.mode = 'max'
        self.pad = pad
        self.stride = stride
        
        self.dWeight = None
        self.arg_max = None
        self.row_mean = None
        
    def forward(self,input):
        N,C,H,W = input.shape
        OH = ((H-self.pool_h)//self.stride)+1
        OW = ((W-self.pool_w)//self.stride)+1
        
        flatten_img = common_functions.im2col(input, self.pool_h, self.pool_w, self.stride, self.pad)
        flatten_img = flatten_img.reshape(-1, self.pool_h * self.pool_w)
        
        if(self.mode == 'max'):
            out = np.max(flatten_img, axis = 1)
        elif(self.mode == 'average'):
            out = np.mean(flatten_img, axis = 1)
            self.row_mean = out.shape[1]
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return out
    
    #def backward(self, dout):
    #    
    #    if(self.mode == 'max'):
    #    elif(self.mode == 'average'):
    #        (1 / self.row_avg) * dout
    #    
    #    return dx
    
class sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class softmax:
    def __init__(self):
        self.out = None

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Overflow ¹æÁö
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        return dx
#class tanh:
#    return

#class ReLU:
#    return np.maximum(0,x)
class flatten:
    def __init__(self):
        self.original_shape = None
    
    def forward(self, input):
        self.original_shape = input.shape 
        out = input.reshape(input.shape[0],-1)
        return out
    
    #def backward(self, dout):
           
