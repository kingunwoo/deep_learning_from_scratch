# -*- coding: utf-8 -*-

import numpy as np
from model import LeNet_v1
from common_functions import load_mnist

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 데이터를 로드하는 함수에서 flatten=False를 명시적으로 사용하여 4차원 데이터를 가져옴
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, one_hot_label=False)
print(x_test.shape)
# 테스트 데이터를 차원 변환
x_test = x_test.reshape(-1, 1, 28, 28)

# 모델 초기화
model = LeNet_v1(input=x_test[:10], target=t_test[:10])

# 순전파 실행 및 각 레이어의 출력 확인
output = x_test[:10]
for layer_name, layer in model.layers_dict.items():
    output = layer.forward(output)
    print(f"Output of {layer_name}: {output.shape}")

# 최종 출력 결과 확인
print("Final output before softmax:", output)

# softmax 적용
output_prob = softmax(output)
print("Final output after softmax:", output_prob)

# 각 데이터 포인트에 대한 예측 클래스
predicted_class = np.argmax(output_prob, axis=1)
print("Predicted class for each sample:", predicted_class)
