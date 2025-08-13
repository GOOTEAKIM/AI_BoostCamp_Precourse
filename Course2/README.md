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

## Day 5 : PyTorch Dataset

- 모델에 데이터를 먹이는 방법

### Dataset 클래스

- 데이터 입력 형태를 정의하는 클래스
- 데이터를 입력하는 방식의 표준화
- Image, Text, Audio 등에 따른 다른 입력 정의

#### Dataset 클래스 생성시 유의점

- 데이터 형태에 따라 각 함수를 다르게 정의
- 모든 것을 데이터 생성 시점에 처리할 필요는 없음
  - > image의 Tensor 변화는 학습에 필요한 시점에 변환
- 데이터 셋에 대한 표준화된 처리방법 제공 필요
  - > 후속 연구자 또는 동료에게는 빛과 같은 존재
- 최근에는 HuggingFace 등 표준화된 라이브러리 사용

### DataLoader 클래스

- Data의 Batch를 생성해주는 클래스
- 학습직전 (GPU feed전) 데이터의 변환을 책임
- Tensor로 변환 + Batch 처리가 메인 업무
- 병렬적인 데이터 전처리 코드의 고민 필요

  ```python
  DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
      batch_sampler=None, num_workers=0, collate_fn=None,
      pin_memory=False, drop_last=False, timeout=0,
      worker_init_fn=None, *, prefetch_factor=2,
      persistent_workers=False)
  ```

## Day 6 : 모델 불러오기

### model.save()

- 학습의 결과를 저장하기 위한 함수
- 모델 형태 (architecture) 와 파라미터를 저장
- 모델 학습 중간 과정의 저장을 통해 최선의 결과 모델을 선택
- 만들어진 모델을 외부 연구자와 공유하여 학습 재연성 향상

### checkpoints

- 학습의 중간 결과를 저장하여 최선의 결과를 선택
- earlystopping 기법 사용시 이전 학습의 결과물을 저장
- loss와 metric 값을 지속적으로 확인 저장
- 일반적으로 epoch, loss, metric을 함께 저장하여 확인
- colab에서 지속적인 학습을 위해 필요

### Transfer learning (전이학습)

- 다른 데이터셋으로 만든 모델을 현재 데이터에 적용
- 일반적으로 대용량데이터셋으로 만들어진 모델의 성능 증가
- 현재 DL 에서는 가장 일반적인 학습 기법
- backbone architecture 가 잘 학습된 모델에서 일부분만 변경하여 학습을 수행

## Day 7 : monitoring tools

### Tensorboard

- TensorFlow의 프로젝트로 만들어진 시각화 도구
- 학습 그래프, metric, 학습 결과의 시각화 지원
- PyTorch도 연결 가능 -> DL 시각화 핵심 도구
 
- scalar : metric 등 상수 값의 연속(epoch)을 표시
- graph : 모델의 computational graph 표시
- histogram : weight 등 값의 분포를 표현
- Image : 예측 값과 실제 값을 비교 표시
- mesh : 3d 형태의 데이터를 표현하는 도구

  ```python
  import os
  import torch
  from tensorboard import notebook

  from torch.utils.tensorboard import SummaryWriter # 기록 생성 객체 SummaryWriter 생성 
  import numpy as np

  # 로그를 저장할 기본 디렉토리를 설정합니다.
  logs_base_dir = "logs"
  os.makedirs(logs_base_dir, exist_ok=True) # Tensorboard 기록을 위한 directory 생성

  writer = SummaryWriter(exp)

  # 임의의 손실 및 정확도 값을 로그에 기록합니다.
  for n_iter in range(100):
      writer.add_scalar('Loss/train', np.random.random(), n_iter)
      writer.add_scalar('Loss/test', np.random.random(), n_iter)
      writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
      writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
  writer.flush() # 값 기록(disk에 쓰기)

  """
  add_scalar 함수 :scalar 값을 기록
  Loss / train : loss category에 train값
  n_iter : x축의 값
  """
  ```

#### 개발환경 설정

  ```bash
  conda install -c conda-forge tensorboard
  ```

### weight & biases

- 머신러닝 실험을 원활히 지원하기 위한 상용도구
- 협업, code versioning, 실험 겨과 기록 등 제공
- MLOps의 대표젹인 툴로 저변 확대 중

## Day 8 : Multi GPU 학습

### 개념정리

- Single vs Multi
- GPU vs Node
- Single Node Single GPU
- Single Node Multi GPU
- Multi Node Multi GPU

### Model parallel

- 다중 GPU에 학습을 분산하는 두가지 방법
  - 모델 나누기 / 데이터 나누기
- 모델을 나누는 것은 생각보다 예전에 썼음
- 모델의 병목, 파이프라인의 어려움으로 인해 모델 병렬화는 고난이도 과제

  ```python
  class ModelParallelResNet50(ResNet):
      def __init__(self, *args, **kwargs):
          super(ModelParallelResNet50,self).__init_(
              Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)
          
          self.seq1 = nn.Sequential(
              self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2).to('cuda:0') # 첫번째 모델을 cuda 0에 할당

          self.seq2 = nn.Sequential(
              self.layer3, self.layer4, self.avgpool,).to('cuda:1') # 두번째 모델을 cuda 1에 할당 

          self.fc.to('cuda:1')

      def forward(self, x): 
          x = self.seq2(self.seq1(x).to('cuda:1')) # 두 모델을 연결
          return self.fc(x.view(x.size(0), -1))
  ```

### Data parallel

- 데이터를 나눠 GPU에 할당 후 결과의 평균을 취하는 방법
- minibatch 수식과 유사한데 한번에 여러 GPU에서 수행
- PyTorch에서는 2가지 방식 제공
  - DataParallel
    ```python
    parallel_model = torch.nn.DataParallel(model) # Encapsulate the model

    predictions = parallel_model(inputs). # Forward pass on multi-GPUs
    loss = loss_function(predictions, labels) # Compute loss function
    loss.mean().backward() # Average GPU-losses +backward pass
    optimizer.step() # Optimizer step
    predictions = parallel_model(inputs) # Forward pass with new parameters
    ```

    - 단순히 데이터를 분배한 후 평균을 취한다
    - > GPU 사용 불균형 문제 발생, Batch 사이즈 감소 (한 GPU가 벼목), GIL



  - DistributedDataParallel
    - 각 CPU 마다 process 생성하여 개별 GPU에 할당
    - > 기본적으로 DataParallel로 하나 개별적으로 연산의 평균을 냄

      ```python
      train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) # 샘플러 사용
      shuffle = False
      pin_memory = True

      trainloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True, pin_memory=pin_memory, num_workers=3, shuffle=shuffle, sampler=train_sampler)
      ```

      ```python
      from multiprocessing import Pool 

      def f(x): # Python의 멀티프로세싱 코드
          return x*x

      if __name__ == '__main ':
          with Pool(5) as p:
          print(p.map(f, [1, 2, 3]))
      ```

      ```python
      def main():
          n_gpus = torch.cuda.device_count()
          torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, ))

      def main_worker(gpu, n_gpus):
          image_size = 224
          batch_size = 512
          num_worker = 8
          epochs = ...

          batch_size = int(batch_size / n_gpus)
          num_worker = int(num_worker / n_gpus)
          torch.distributed.init_process_group(
          backend='nccl’, init_method='tcp://127.0.0.1:2568’, world_size=n_gpus, rank=gpu) # 멀티프로세싱 통신 규약 정의

          model = MODEL
          torch.cuda.set_device(gpu)
          model = model.cuda(gpu)
          model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu]) # Distributed DataParallel 정의
      ```
