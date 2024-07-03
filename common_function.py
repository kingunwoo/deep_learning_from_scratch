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

import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # 입력 데이터에 패딩을 추가합니다.
    input_data_padded = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    
    # 입력 데이터의 형상을 가져옵니다.
    N, C, H, W = input_data.shape
    
    # 출력 높이와 너비를 계산합니다.
    OH = (H + 2 * pad - filter_h) // stride + 1
    OW = (W + 2 * pad - filter_w) // stride + 1
    
    # 출력 행렬의 크기를 계산합니다.
    out_matrix_col_len = filter_h * filter_w * C  # 채널을 고려하여 크기 계산
    out_matrix_row_len = OH * OW * N
    out_matrix = np.zeros((out_matrix_row_len, out_matrix_col_len))
    
    # 2중 for문을 사용하여 입력 데이터를 필터 크기만큼 슬라이딩하면서 변환합니다.
    position_count = 0
    for i in range(OH):
        for j in range(OW):
            # 슬라이딩 윈도우를 각 위치로 이동시키며 평탄화
            patch = input_data_padded[:, :, i*stride:i*stride+filter_h, j*stride:j*stride+filter_w]
            out_matrix[position_count:position_count + N] = patch.reshape(N, -1)
            position_count += N
    
    return out_matrix

# 테스트를 위한 예시
input_data = np.random.rand(1, 1, 5, 5)  # 1x1 채널의 5x5 크기의 데이터
filter_h, filter_w = 3, 3
output = im2col(input_data, filter_h, filter_w, stride=1, pad=1)
print(output)



def get_data(): 
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test