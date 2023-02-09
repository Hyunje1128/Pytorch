#1

import torch

#2 텐서는 파이토치에서 다양한 수식을 계산하는 데 사용하는 가장 기본적인 자료구조이다.
#  수학의 벡터나 행렬을 일반화한 개념으로서, 숫자들을 특정한 모양으로 배열한 것
#  탠서에는 '차원' 또는 '랭크'라는 개념이 있다.
#  1 -> 스칼라, 모양은[]
#  [1,2,3] -> 벡터, 모양은 [3]
#  [[1,2,3]] -> 행렬, 모양은 [1,3]
#  [[[1,2,3]]] -> n랭크 텐서, 모양은 [1,1,3]

x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

# print(x)
# print("Size:", x.size())
# print("Shape:", x.shape)
# print("랭크(차원):", x.ndimension())

# 랭크 늘리기 (unsqueeze)                     #3

# x = torch.unsqueeze(x, 0)
# print(x)
# print("Size:", x.size())
# print("Shape:", x.shape)
# print("랭크(차원):", x.ndimension())

# 랭크 줄이기 (squeeze)                       #4

# x = torch.squeeze(x)
# print(x)
# print("Size:", x.size())
# print("Shape:", x.shape)
# print("랭크(차원):", x.ndimension())

# view                                      #5

# x = x.view(9)
# print(x)
# print("Size:", x.size())
# print("Shape:", x.shape)
# print("랭크(차원):", x.ndimension())

try:                                        #6
    x = x.view(2,4)
except Exception as e:
    print(e)