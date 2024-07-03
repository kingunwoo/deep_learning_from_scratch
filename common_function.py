import sys,os
sys.path.append(os.pardir)
from mnist_dataset.mnist import load_mnist
# It takes a long time when you run the mnist.py file for the first time, 
# but it doesn't take long because you import 
# the local pickle from the second time
import numpy as np
import pickle
import optimizer 
import loss_function

dim4_arr = np.ones((2,3,4,4))

def im2col(input_data, filter_h, filter_w, stride = 1, pad = 0):
    #input_data is 4 dimension
    one_input_data = input_data[0][0]
    #zero th data,zero th chanel
    OH = (one_input_data.shape[0] + (2*pad) - filter_h)//stride + 1
    OW = (one_input_data.shape[1]+(2*pad) - filter_w)//stride + 1
    out_matrix_col_len = filter_h * filter_w
    out_matrix_row_len = OH * OW
    out_matrix = np.zeros(out_matrix_row_len, out_matrix_col_len)
    
    flatten = np.reshape(input_data,(-1,one_input_data.shape[0],one_input_data.shape[1]))
    #4dim -> 3dim
    
    position_count = 0 
    for mat in flatten:
        h_pos = 0
        for h in OH:
            w_pos = 0
            for w in OW:
                out_matrix[position_count]= mat[h_pos:h_pos+filter_h, w_pos:w_pos+filter_w].flatten()
                position_count += 1
    
    return out_matrix

def get_data(): 
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test