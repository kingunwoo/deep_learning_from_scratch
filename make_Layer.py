import numpy as np
import activation_functions
class fc_layer:
    def __init__(self,input_data,length_output_node,activation): 
        self.output = np.ones_like(length_output_node)
        self.input = input_data.reshape(-1,784)
        self.activation = activation
        if(activation == activation_functions.sigmoid or activation_functions.tanh):
            # Xavier 초기값을 사용할려고 함,편향은 weight 0번 인덱스
            std = np.sqrt(1/(len(input)+len(output)))
            self.weight = np.random.randn()
        elif(activation == activation_functions.ReLU):
            #He 초기값을 사용할려고 함
            
    def forward(self):
            
    
                