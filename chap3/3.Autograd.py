import torch

#1 값이 1.0인 스칼라 텐서 w를 정의하고, 수식을 w에 대해 미분하여 기울기를 계산하자.
# w의 requires_grad를 True로 설정하면, 파이토치의 Autograd 기능이 자동으로 계산할 때 w에 대한 미분값을 w.grad에 저장합니다.

w = torch.tensor(1.0, requires_grad=True)

#2 a = w x 3

a = w*3

#3 l = a ^ 2

l = a**2

#4 l을 w로 미분하려면 연쇄법칙을 이용

l.backward()
print('1을 w로 미분한 값은 {}'.format(w.grad))



