import numpy as np

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

def col2im(col, input_shape,  filter_h,  filter_w, stride=1, pad=0):
    
    return img

def get_data(): 
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test