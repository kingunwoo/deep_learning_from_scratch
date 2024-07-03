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

import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # 입력 데이터에 패딩을 추가합니다.
    input_data_padded = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    
    # 하나의 입력 데이터와 채널을 선택합니다.
    one_input_data = input_data_padded[0][0]
    
    # 출력 높이와 너비를 계산합니다.
    OH = (one_input_data.shape[0] - filter_h) // stride + 1
    OW = (one_input_data.shape[1] - filter_w) // stride + 1
    
    # 출력 행렬의 크기를 계산합니다.
    out_matrix_col_len = filter_h * filter_w * input_data.shape[1]  # 채널을 고려하여 크기 계산
    out_matrix_row_len = OH * OW * input_data.shape[0]
    out_matrix = np.zeros((out_matrix_row_len, out_matrix_col_len))
    
    # 입력 데이터를 평탄화합니다.
    flatten = np.reshape(input_data_padded, (-1, one_input_data.shape[0], one_input_data.shape[1]))
    
    # 위치 카운터 초기화
    position_count = 0
    
    # 입력 데이터에 대한 반복문
    for mat in flatten:
        for h in range(0, one_input_data.shape[0] - filter_h + 1, stride):
            for w in range(0, one_input_data.shape[1] - filter_w + 1, stride):
                out_matrix[position_count] = mat[:, h:h+filter_h, w:w+filter_w].flatten()
                position_count += 1
    
    return out_matrix


def get_data(): 
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test