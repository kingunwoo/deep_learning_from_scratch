import numpy as np
import activation_functions
class fc_layer:
    def __init__(self,input_data,length_output_node,activation): 
        self.output = np.ones(length_output_node)
        self.input = input_data.reshape(-1,784)
        self.activation = activation
        if(activation == activation_functions.sigmoid or activation == activation_functions.tanh):
            # using Xavier initialization 
            std = np.sqrt(1/len(input))
            self.weight = np.random.randn(input.shape[0], output.shape) *std
        elif(activation == activation_functions.ReLU):
            #using He initialization
            std = np.sqrt(2/len(input))
            self.weight = np.random.randn(input.shape[0], output.shape) *std
        self.bias = np.ones_like(length_output_node)
        
    def forward(self):
        output = np.dot(input,weight) + bias
        
    def backward(self)
        
    
                