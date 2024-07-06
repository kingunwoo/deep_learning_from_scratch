import numpy as np
from model import LeNet_v1
from common_functions import load_mnist

# 데이터를 로드하는 함수에서 flatten=False를 명시적으로 사용하여 4차원 데이터를 가져옴
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, one_hot_label=False)
print(x_test.shape)
# 테스트 데이터를 차원 변환
x_test = x_test.reshape(-1, 1, 28, 28)  # x_test를 (N, C, H, W) 형태로 조정

# 모델 초기화
model = LeNet_v1(input=x_test[:10], target=t_test[:10])  # 첫 10개의 데이터 사용

# 순전파 실행 및 각 레이어의 출력 확인
output = x_test[:10]
for layer_name, layer in model.layers_dict.items():
    output = layer.forward(output)
    print(f"Output of {layer_name}: {output.shape}")

# 마지막 출력 결과 확인
print("Final output:", output)
