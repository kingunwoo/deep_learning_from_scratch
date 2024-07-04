import numpy as np
import sys,os
sys.path.append(os.pardir)
from mnist_dataset.mnist import load_mnist

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    input_data_padded = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    N, C, H, W = input_data.shape
    
    OH = (H + 2 * pad - filter_h) // stride + 1
    OW = (W + 2 * pad - filter_w) // stride + 1
    
    out_matrix_col_len = filter_h * filter_w * C 
    out_matrix_row_len = OH * OW * N
    out_matrix = np.zeros((out_matrix_row_len, out_matrix_col_len))
    
    position_count = 0
    for i in range(OH):
        for j in range(OW):
            patch = input_data_padded[:, :, i*stride:i*stride+filter_h, j*stride:j*stride+filter_w]
            out_matrix[position_count:position_count + N] = patch.reshape(N, -1)
            position_count += N
    return out_matrix


def col2im(cols, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    out_h = (H_padded - filter_h) // stride + 1
    out_w = (W_padded - filter_w) // stride + 1

    img = np.zeros((N, C, H_padded, W_padded))
    col_reshaped = cols.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    overlapping_count = np.zeros_like(img)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col_reshaped[:, :, y, x, :, :]
            overlapping_count[:, :, y:y_max:stride, x:x_max:stride] += 1
    if pad > 0:
        img = img[:, :, pad:-pad, pad:-pad]
    
    # dividing overrapping region
    img = img / overlapping_count
    return img


def get_data(): 
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test