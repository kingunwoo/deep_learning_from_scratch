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
    # �Է� �����Ϳ� �е��� �߰��մϴ�.
    input_data_padded = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    
    # �Է� �������� ������ �����ɴϴ�.
    N, C, H, W = input_data.shape
    
    # ��� ���̿� �ʺ� ����մϴ�.
    OH = (H + 2 * pad - filter_h) // stride + 1
    OW = (W + 2 * pad - filter_w) // stride + 1
    
    # ��� ����� ũ�⸦ ����մϴ�.
    out_matrix_col_len = filter_h * filter_w * C  # ä���� ����Ͽ� ũ�� ���
    out_matrix_row_len = OH * OW * N
    out_matrix = np.zeros((out_matrix_row_len, out_matrix_col_len))
    
    # 2�� for���� ����Ͽ� �Է� �����͸� ���� ũ�⸸ŭ �����̵��ϸ鼭 ��ȯ�մϴ�.
    position_count = 0
    for i in range(OH):
        for j in range(OW):
            # �����̵� �����츦 �� ��ġ�� �̵���Ű�� ��źȭ
            patch = input_data_padded[:, :, i*stride:i*stride+filter_h, j*stride:j*stride+filter_w]
            out_matrix[position_count:position_count + N] = patch.reshape(N, -1)
            position_count += N
    
    return out_matrix

# �׽�Ʈ�� ���� ����
input_data = np.random.rand(1, 1, 5, 5)  # 1x1 ä���� 5x5 ũ���� ������
filter_h, filter_w = 3, 3
output = im2col(input_data, filter_h, filter_w, stride=1, pad=1)
print(output)



def get_data(): 
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test