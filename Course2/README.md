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