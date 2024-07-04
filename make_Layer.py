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
        flatten_img = common_functions.im2col(input_data, self.filter_h, self.filter_w, self.stride, self.pad)
        filter.reshape(-1, self.filter_n)
        flatten_conv_img = np.dot(flatten_img,filter)
        output_img = common_functions.col2im(flatten_conv_img, input_data.shape, self.filter_h, self.filter_w, self.stride, self.pad)
        return output_img
        
    def backward(self):
        return 

class fc:
    def __init__(self,input_data,length_output_node,activation): 
        self.output = np.ones(length_output_node)
        self.input = input_data.reshape(-1,784)
        self.activation = activation
        if(activation == activation_functions.sigmoid or activation == activation_functions.tanh):
        else:
            gfg
    def forward(self):
        output = np.dot(input,weight) + bias
        return output        
    def backward(self):
        return
    
class pooling:
    def __init__(self, pool_h, pool_w,  mode='max', pad = 0, stride = 1):
    
        
if __name__ == "__main__":
    import common_functions  # common_functions 모듈 가져오기

    input_data = np.random.randn(1, 1, 28, 28)  # 예시 입력 데이터
    conv_layer = convolution(filter_n=6, filter_h=5, filter_w=5, pad=2, stride=1, activation='sigmoid')
    output_data = conv_layer.forward(input_data)
    print(output_data.shape)