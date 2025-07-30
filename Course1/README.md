# 인공지능 기초 다지기 (1)

## Day 0 : File System & Terminal Basic

### 1. 컴퓨터 OS (Operating System, 운영체제)

- 프로그램이 동작할 수 있는 구동 환경

### 2. 파일 시스템 (File System)

- OS에서 파일을 저장하는 **트리구조** 저장체계
- 컴퓨터 등의 기기에서 의미 있는 정보를 담는 논리적인 단위
- 모든 프로그램은 파일로 구성되어 있고, 파일을 사용한다.

- **파일의 기본 체계 - 파일 VS 디렉토리**
  
  - 디렉토리 (Directory)
    - 폴더 또는 디렉토리로 불림
    - 파일과 다른 디렉토리를 포함할 수 있음
  
  - 파일 (File)
    - 컴퓨터에서 정보를 저장하는 논리적인 단위
    - 파일은 파일명과 확장자로 식별 됨 (ex : hello.py)
    - 실행, 쓰기, 읽기 등을 할 수 있음

- **절대 경로와 상대 경로**
  - 경로 : 컴퓨터 파일의 고유한 위치, 트리구조상 노드의 연결

  - 절대 경로 : 루트 디렉토리부터 파일까지의 경로
    - ex) C:\Users\user\Desktop\AI_BoostCamp_Precourse

  - 상대 경로 - 현재 있는 디렉토리부터 타깃 파일까지의 경로
    - ex) ../../AI_BoostCamp_Precourse


### 3. 터미널

- 마우스가 아닌 키보드로 명령을 입력 프로그램 실행

- **CLI (Command Line Interface)**
  - GUI와 달리 Text를 사용하여 컴퓨터에 명령을 입력하는 인터페이스 체계
  - Windows - CMD window, Window Terminal
  - Mac, Linux - Terminal

- Console = Terminal = CMD 창
  - 어원 : 디스틀레이와 키보드가 조합된 장치
  - 현재 : CLI로 입력하는 화면

- **기본 명령어**
  - 각 터미널에서는 프로그램을 작동하는 shell이 존재
    - > shell 마다 다른 명령어를 사용

    | 윈도우 CMD창 명령어 | shell 명령어 | 설명 |
    |-|-|-|
    | CD | cd | 현재 디렉토리 이름을 보여주거나 바꿈 |
    | CLS | clear |  CMD 화면에 표시된 것을 모두 지움 | 
    | COPY | cp | 하나 이상의 파일을 다른 위치로 복사 |
    | DEL | rm | 하나 이상의 파일을 지움 |
    | DIR | ls | 디렉토리에 있는 파일과 하위 디렉토리 목록을 보여줌 |

## Day 1 : Python 개요

### Python의 시작
  - 귀도 반 로섬이 발표
  - **독립적인 플랫폼**
  - **인터프리터 언어**
  - **객체 지향**
  - **동적 타이핑 언어**
  - C 언어로 구현되었음

### Python의 특징

- 플랫폼 = OS : 윈도우, 리눅스, 안드로이드, 맥 OS, IOS 등 프로그램이 실행되는 **운영 체제**

- **독립적인 = 관계없는, 상관없는** : OS에 상관없이 한번 프로그램을 작성하면 사용가능

- **인터프리터 = 통역기를 사용하는 언어** : 소스코드를 바로 실행할 수 있게 지원하는 프로그램 실행 방법

| 항목 | 컴파일러 | 인터프리터 |
| - | - | - |
| 작동방식 | 소스코드를 기계어로 먼저 번역하여 해당 플랫폼에 최적화된 프로그램을 실행 | 별도의 번역 과정 없이 소스코드를 실행 시점에 해석하여 컴퓨터가 처리할 수 있도록 함 |
| 장점/단점 | 실행속도가 빠름, 한 번에 많은 기억장소 필요 | 간단히 작성, 메모리가 적게 필요, 실행속도가 느림 |
| 주요 언어 | C, Java, C++, C# | python, 스칼라 |

- **객체 지향적 언어 : 싫행 순서가 아닌 단위 모듈 (객체) 중심으로 프로그램을 작성**
  - 하나의 개체는 어떤 목적을 달성하기 위한 행동 (method)와 속성 (attribute)을 가지고 있음

- **동적 타이핑 언어** : 프로그램이 **실행하는 시점에 프로그램**이 사용해야할 **데이터에 대한 타입을 결정** 

## Day 2 : 개발환경

- 프로그램을 작성하고, 실행시키는 환경
- 일반적으로 **코딩 환경** 이라고 부름
- 개발환경을 결정

1) OS
2) Python Interpreter
3) 코드 편집기 (Editor)

### OS

- 선호하는 운영체제를 선정

- Windows 
  - 친숙함, 초기엔 쉬움
  - 모듈 설치 어려움, 참고 문서 부족

- Linux
  - 모듈 설치 쉬움
  - 무료, 참고문서 많음
  - OS 자체 사용이 어려움

- MacOS
  - 모듈 설치 쉬움
  - 참고문서도 많음
  - 비쌈

### Python Interpreter

- 2.7과 3.x 버전이 존재 : 현재는 3.x 기준으로 모든 시스템이 작동됨
- 기존 라이브러리 사용 여부에 따라 버전을 선택

### 코드 편집기

- text 타입의 문서를 ㅈ장하는 모든 편집기 사용가능
  - 메모장
  - VI editor
  - Sublime Text
  - Atom
  - VS Code
  - PyCharm

- **Type 1 - Python**
  - Anaconda + VS Code

- **Type 2- Data analysis + Deep learning**
  - Jupyter + colab

## Day 3 : Anaconda 설치

- 설치 링크 : https://www.anaconda.com/download/success
- windows 최신 버전 (python 3.13)
- 3.9.13이 되도록 변경
- anaconda prompt에서 **conda init powershell** 실행
- powershell에서 base가 자동 실행됨
  - command로 방지
      ```bash
      conda config --set auto_activate_base false
      ```
  - 해결되었는지 확인

      ```bash
      conda config --show | findstr auto_activate
      auto_activate: false 로 나오면 정상
      ```

- 3.9.13 버전으로 가상환경 만들기

  ```bash
  conda create -n py3913 python=3.9.13
  conda activate py3913
  python --version
  ```

## Day 5 : Jupyter Notebook, Colab

### Jupyter Notebook

- IPython 커널을 기반으로 한 대화형 python shell
- 일반적인 terminal shell + 웹 기반 데이터 분석 Notebook 제공
- 미디어, 텍스트, 코드, 수식 등을 하나의 문서로 표현 가능
- 사실상의 데이터 분석 Interactive Shell의 표준

- (base) PS C:\Users\user> cd workspace 에서 jupyter notebook를 실행한다

  ```bash
  conda activate base

  cd workspace

  jupyter notebook 
  ```

- 단축키

  | 기능 | 단축키 |
  | - | - |
  | 실행 | `Ctrl + Enter`, `Shift + Enter`, `Alt + Enter` |
  | 툴팁 표시 | `Shift + Tab` |
  | 들여쓰기 | `Ctrl + ]` or `Ctrl + [` |
  | 셀 나누기 | `Ctrl + Shift + -` |
  | 아래 셀이랑 합치기 | `Shift + M` |
  | 셀 오려두기 | `X` |
  | 셀 복사 | `C` |
  | 셀 붙여넣기 | `V` or `Shift + V` |
  | 셀 지우기 | `DD` |
  | 셀 지우기 취소 | `Z` |
  | Markdown 변환 | `MM` |
  | Code로 변환 | `YY` |

### Colab

- 구글이 개발한 클라우드 기반의 jupyter notebook
- 구글 드라이브 + GCP + jupyter
- 초반 여러가지 모듈 설치의 장점을 가짐
- 구글 드라이브 파일을 업로드하여 사용가능
- VS Code와 연결해서 사용 가능
- GPU 무료 사용 가능

## Day 6 : Python - Variable & List

### Variable (변수)

- 가장 기초적인 프로그래밍 문법 개념
- 데이터 (값)을 저장하기 위한 메모리 공간의 프로그래밍상 이름 == **값을 저장하는 장소**

- 변수는 **메모리 주소**를 가지고 있고 변수에 들어가는 **값**은 **메모리 주소**에 할당됨

#### 컴퓨터의 구조 - 폰 노이만 아키텍처

1. 사용자가 컴퓨터에 값을 입력하거나 프로그램을 실행 
2. 그 정보를 먼저 메모리에 저장
3. CPU가 순차적으로 그 정보를 해석, 계산
4. 사용자에게 결과값 전달

#### basic operations (간단한 연산)

1. 기본 자료형
2. 연산자와 피연산자
3. 데이터 형변환

- **기본 자료형 (primitive data types)**

  - int
  - float
  - string
  - boolean

  - **Dynamic Typing**
    - 코드 실행 시점에 데이터의 Type을 결정 (python의 장점)

- **연산자와 피연산자**

  - +,-,*,/
  - 수식에서 여산자의 역할은 수학에서 연산자와 동일
  - 연산의 순서는 수학에서 연산 순서와 같음
  - **문자간에도 + 연산 가능**

  - 데이터 형변환
    - 정수형 <> 실수형
      - float() 와 int() 사용하여 형변환 가능

    - 숫자 <> 문자열
      - 문자열로 선언된 값도 int(), float() 함수로 형 변환 가능

    - type() : 변수의 데이터 형을 확인

### list

- 시퀀스 자료형, 여러 데이터들의 집합
- int, float 같은 다양한 데이터 타입 포함

- 특징
  - indexing
  - slicing
  - 리스트 연산
  - 추가 삭제
  - 메모리 저장 방식
  - 패킹과 언패킹
  - 2차원 리스트

- indexing
  - list에 있는 값들은 **주소(offset)**를 가짐
    - > 주소를 사용해 할당된 값을 호출

- slicing
  - list의 값들을 잘라서 쓰는 것
  - list의 주소 값을 기반으로 부분 값 반환

- 리스트의 연산
  - concatenation, is_in, 연산 함수들

- 추가, 삭제
  - append, extend, insert, remove, del

- 패킹과 언패킹
  - 패킹 : 한 변수에 여러 개의 데이터를 넣는 것
  - 언패킹 : 한 변수의 데이터를 각각의 변수로 변환

- 2차원 리스트
  - 리스트 안에 리스트를 만들어 행렬 생성 

#### python list만의 특징

- 다양한 Data Type이 하나의 list에 들어간다
  - 리스트에 문자열, 숫자, 리스트 상관없이 넣을 수 있다

