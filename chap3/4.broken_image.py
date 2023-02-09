#대부분의 프로그래머는 다음과 같이 문제에 접근

#1 weird_function()함수의 소스코드를 분석한다.
#2 분석을 토대로 weird_function()함수의 동작을 반대로 이행하는 함수를 구현한다.
#3 2에서 구현한 함수에 오염된 이미지를 입력해서 복구된 이미지를 출력한다.

#우리가 택할 해결책은 좀 더 머신러닝과 수학적 최적화에 가까운 방법이다.

#1 오염된 이미지와 크기가 같은 랜덤 텐서를 생성한다.
#2 랜덤 텐서를 weird_function()함수에 입력해 똑같이 오염된 이미지를 가설이라고 부른다.
##a [사실] 원본 이미지가 weird_function() 함수에 입력되어 오염된 이미지를 출력했다.
##b [사실] 인위적으로 생성한 무작위 이미지가 weird_function() 함수에 입력되어 가설을 출력했다.
#3 가설과 오염된 이미지가 같다면, 무작위 이미지와 원본 이미지도 같을 것이다.
#4 그러므로 weird_function(random_tensor) = broken_image 관계가 성립하도록 만든다.

#문제 해결과 코드 구현

#1 그다음 파이토치를 임포트합니다. 그리고 오염된 이미지와 복원된 이미지를 출력하는데 맷플롯립 라이브러리를 이용하겠습니다.
#  맷플롯립 라이브러리에서 플롯을 생성해주는 pyplot 모듈을 plt로 이름지어 임포트합니다.(matplotlib.pyplot -> plt)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import pickle
import matplotlib.pyplot as plt

#2 오염된 이미지를 파이토치 텐서의 형태로 읽은 후, 어떻게 보이는지 확인해보겠습니다. 컴퓨터는 이미지를 픽셀값들을 숫자로 늘어놓은 행렬로 표현됩니다.
#  앞으로 우리가 복원해야 할 이미지인 broken_image 역시 이미지 행렬을 랭크 1의 벡터로 표현한 텐서 데이터입니다.
#  10,000개의 원소를 지닌 [100, 100] 모양의 행렬이 [10000] 모양의 벡터로 표현된 형식입니다. 맷플롯립을 이용해 [100, 100]의 형태로 바꾼 후 시각화하자.

shp_original_img = (100, 100)
broken_image = torch.FloatTensor(pickle.load(open("./broken_image_t.p", 'rb'), encoding='latin1'))

plt.imshow(broken_image.view(100,100))
plt.show()

#3 이미지를 오염시키는 weird_function()함수의 코드는 다음과 같습니다.
#  코드가 복잡해 보이겠지만 걱정마세요. 이를 이해할 필요는 없습니다.
#  아래의 코드를 이해하고 반대로 실행시키기보다는 머신러닝을 이용하여 복원할 것이기 때문이다.

def weird_function(x, n_iter=5):
    h = x
    filt = torch.tensor([-1./3, 1./3, -1./3])
    for i in range(n_iter):
        zero_tensor = torch.tensor([1.0*0])
        h_l = torch.cat((zero_tensor, h[:-1]),0)
        h_r = torch.cat((h[1:], zero_tensor), 0)
        h = filt[0] * h + filt[2] * h_l + filt[1] * h_r
        if i % 2 == 0:
            h = torch.cat((h[h.shape[0]//2:],h[:h.shape[0]//2]), 0)
    return h

#4 다음으로 무작위 텐서를 weird_function() 함수에 입력해 얻은 가설 텐서와 오염된 이미지 사이의 오차를 구하는 함수를 구현

def distance_loss(hypothesis, broken_image):
    return torch.dist(hypothesis, broken_image)

#5 다음으로 무작위 값을 가진 텐서를 생성합니다. 이 텐서는 경사하강법을 통해 언젠가는 원본 이미지의 형상을 하게 될 겁니다.
#  이 무작위 텐서 역시 broken_image와 같은 모양과 랭크를 지녀야 합니다. 즉, [100, 100] 모양의 행렬이 [10000] 모양의 벡터로 표현된 텐서입니다.

random_tensor = torch.randn(10000, dtype = torch.float)

#6 경사하강법은 여러 번 반복해서 이뤄집니다. 이때 한 반복에서 최솟점으로 얼마나 많이 이동하는지,
#  즉 학습을 얼마나 급하게 진행하는가를 정하는 매개변수를 학습률이라고 합니다.

lr = 0.8

#7 이제 경사하강법에 필요한 준비가 모두 끝났습니다. 본격적으로 경사하강법의 몸체인 for 반복문을 구현
#  먼저, 오차 함수를 random_tensor로 미분해야 하니 requires_grad를 True로 설정

for i in range(0, 50000):
    random_tensor.requires_grad_(True)
    hypothesis = weird_function(random_tensor)
    loss = distance_loss(hypothesis, broken_image)
    loss.backward()
    with torch.no_grad():
        random_tensor = random_tensor - lr*random_tensor.requires_grad
    if i % 1000 == 0:
        print('Loss at {} = {}'.format(i, loss.item()))
#8 무작위 텐서를 weird_function() 함수에 통과시켜 가설을 구합니다.
#  그리고 앞에서 정의한 distance_loss() 함수에 hypothesis와 broken_image를 입력해 오차를 계산합니다.
#  그 후 loss.backward() 함수를 호출해 loss를 random_tensor로 미분합니다.



#9 파이토치는 신경망 모델 속 변수들이 지나가는 길인 그래프를 생성합니다. 이번 예제에서는 직접 경사하강법을 구현하기 때문에
#  torch.no_grad() 함수를 이용해 파이토치의 자동 기울기 계산을 비활성화해야 합니다.



plt.imshow(random_tensor.view(100, 100).data)
plt.show()
