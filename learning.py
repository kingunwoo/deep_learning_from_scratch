import numpy as np
from model import LeNet_v1
from common_functions import load_mnist

# �����͸� �ε��ϴ� �Լ����� flatten=False�� ��������� ����Ͽ� 4���� �����͸� ������
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, one_hot_label=False)
print(x_test.shape)
# �׽�Ʈ �����͸� ���� ��ȯ
x_test = x_test.reshape(-1, 1, 28, 28)  # x_test�� (N, C, H, W) ���·� ����

# �� �ʱ�ȭ
model = LeNet_v1(input=x_test[:10], target=t_test[:10])  # ù 10���� ������ ���

# ������ ���� �� �� ���̾��� ��� Ȯ��
output = x_test[:10]
for layer_name, layer in model.layers_dict.items():
    output = layer.forward(output)
    print(f"Output of {layer_name}: {output.shape}")

# ������ ��� ��� Ȯ��
print("Final output:", output)
