import numpy as np

dim4_arr = np.arange(1,55).reshape(2,3,3,3)
one_input_data = dim4_arr[0][0]
dim2_arr =  np.reshape(dim4_arr,(-1,one_input_data.shape[0],one_input_data.shape[1]))
print(dim2_arr)