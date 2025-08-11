# 인공지능 기초 다지기 (2)

## Day 1 : PyTorch 란?

### PyTorch  
- Dynamic Computation Graph
- 연산의 과정을 그래프로 표현
- Define by Run
  - 즉시 확인 가능
  - pythonic code
- 실행을 하면서 그래프를 생성하는 방식
- GPU support, Good API and community
- 사용하기 편하다

- Numpy + AutoGrad + Function
- Numpy 구조를 가지는 Tensor 객체로 array 표현
- 자동 미분을 지원하여 DL 연산을 지원
- 다양한 형태의 DL을 지원하는 함수와 모델을 지원

### TensorFlow  
- Define and run
- 그래프를 먼저 정의 -> 실행 시점에 데이터 feed

## Day 2 : PyTorch Operations

### 개발 환경 설정

- conda 환경에서 설치, ml에 설치 완료

  ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  ```

- vscode 인터프리터를 ml로 설정
- **jupyter notebook의 kernel을 ml로 설정!!**

- 테스트 코드로 사용 가능한지 확인

  ```python
  import torch
  print(torch.__version__)
  print(torch.cuda.is_available())
  ```

### Tensor

- 다차원 Arrays를 표현하는 PyTorch 클래스
- 사실상 numpy의 ndarray
- Tensor를 생성하는 함수도 거의 동일

- Tensor 생성은 list나 ndarray를 사용 가능

- Tensor data types
  - 기본적으로 data 타입은 numpy와 동일

- pytorch의 tensor는 GPU에 올려서 사용가능 -> device

- view : reshape과 동일하게 tensor의 shape을 변환
  - view와 reshape은 contiguity 보장의 차이
- squeeze : 차원의 개수가 1인 차원을 삭제 (압축)
- unsqueeze : 차원의 개수가 1인 차원을 추가

- 기본적인 tensor의 operations은 numpy와 동일
- 행렬곱셈 연산은 함수는 dot이 아닌 mm 사용
- mm과 matmul은 broadcasting 지원 처리

- nn.functional 모듈을 통해 다양한 수식 변환 지원
- PyTorch의 핵심은 **자동 미분의 지원** -> backward 함수 사용

## Day 3 : PyTorch project structure

- 개발 초기 단계에서 대화식 개발 과정이 유리
  - > 학습과정과 디버깅 등 지속적인 확인
- 배포 및 공유 단계에서 notebook 공유가 어려움
  - > 쉬운 재현의 어려움, 실행순서 꼬임
- DL 코드도 하나의 프로그램
  - > 개발 용이성 확보와 유지보수 향상 필요

- 다양한 프로젝트 템플릿이 존재
- 사용자 필요에 따라 수정해서 사용
- 실행, 데이터, 모델, 설정, 로깅, 지표, 유틸리티 등 다양한 모듈들을 분리하여 프로젝트 템플릿화

- 추천 템플릿 : https://github.com/victoresque/pytorch-template

## Day 4 : AutoGrad & Optimizer

### torch.nn.Module

- 딥러닝을 구성하는 Layer의 base class
- Input, Output, Forward, Backword 정의
- 학습의 대상이 되는 parameter (tensor) 정의

### nn.Parameter

- Tensor 객체의 상속 객체
- nn.Module 내에 attribute가 될 때는 required_grad = True 로 지정되어 학습 대상이 되는 Tensor
- 우리가 직접 지정할 일은 잘 없다
  - > 대부분 layer에는 weights 값들이 지정되어 있음

### Backword

- Layer에 있는 Parameter들의 미분 수행
- Forward 의 결과값 (model의 output=예측지) 과 실제값 간의 차이(loss)에 대해 미분을 수행
- 해당 값으로 Parameter 업데이트

  ```python
  for epoch in range(epochs):

    # 이전 기록된 gradient를 0으로 초기화
    optimizer.zero_grad()

    # model을 통해 input forward propagation 진행
    outputs = model(inputs)

    # loss 값 계산
    loss = criterion(outputs, labels)
    print(loss)

    # 모든 파라미터에 대해 gradient 계산
    loss.backward()

    # 파라미터 업데이트
    optimizer.step()
  ```

- 실제 backward는 Module 단계에서 직접 지정가능
- Module에서 backward와 optimizer 오버라이딩
- 사용자가 직접 미분 수식을 써야한느 부담
  - > 쓸 일은 없으나 순서를 이해할 필요는 있음

