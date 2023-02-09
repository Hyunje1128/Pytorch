import torch

#1 행렬곱 구현

# 행렬 w : 정규분포에서 무작위로 값을 뽑아 텐서를 생성하는 randn()함수에 5,3을 인수로 전달하여 5X3의 shape를 가진 텐서를 만듭니다.
# 처음 두 인수는 행과 열의 개수이고, 세 번째 인수는 값의 타입입니다. 무작위로 값을 뽑을 때 실수의 범위 내에서 뽑도록 randn() 함수의 인수 dtpye에 torch.float를 지정합니다.

# 행렬 x : 직접 실수형 원소들을 넣어 3X3의 shape를 가진 텐서를 정의합니다.

w = torch.randn(5,3, dtype=torch.float)
x = torch.tensor([[1.0,2.0], [3.0, 4.0], [5.0, 6.0]])
# print("w size:", w.size())
# print("x size:", x.size())
# print("w:", w)
# print("x:", x)

#2 행렬곱 외에도 다른 행렬 연산에 쓰일 b라는 텐서도 추가로 정의

# b = torch.randn(5,2, dtype=torch.float)
# print("b size:", b.size())
# print("b:", b)

#3 행렬곱은 torch.mm()함수를 사용

wx = torch.mm(w,x)
print("wx size:", wx.size())
print("wx:", wx)

#4 wx 행렬의 원소에 b 행렬의 원소를 더하자

result = wx + b
print("result size:",result.size())
print("result:", result)
