import numpy as np
import activation_functions
class fc_layer:
    def __init__(self,input_data,length_output_node,activation): 
        self.output = np.ones_like(length_output_node)
        self.input = input_data.reshape(-1,784)
        self.activation = activation
        if(activation == activation_functions.sigmoid or activation_functions.tanh):
            # Xavier �ʱⰪ�� ����ҷ��� ��,������ weight 0�� �ε���
            std = np.sqrt(1/(len(input)+len(output)))
            self.weight = np.random.randn()
        elif(activation == activation_functions.ReLU):
            #He �ʱⰪ�� ����ҷ��� ��
            
    def forward(self):
            
    
                