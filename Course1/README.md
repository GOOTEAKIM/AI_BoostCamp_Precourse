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

## Day 7 : Function and Console I/O

### function 

- 어떤 일을 수행하는 코드의 덩어리
- 반복적인 수행을 1회만 작성 후 호출
- 코드를 논리적인 단위로 분리
- 캡슐화 : 인터페이스만 알면 타인의 코드 사용

- 함수 수행 순서

  - 함수 부분을 제외한 메인프로그램부터 시작
  - 함수 호출 시 함수 부분 수행 후 되돌아옴

- parameter : 함수 입력 값 인터페이스

  ```python
  def f(x) :
      return 2 * x + 7
  ```

- argument : 실제 parameter에 대입된 값

  ```python
  print(f(2)) 

  # 11
  ```

- 합수 형태
  - parameter 유무, 반환 값에 따라 함수의 형태가 다름

    | | parameter 없음 | parameter 존재 |
    | - | - | - |
    | 반환 값 없음 | 함수 내의 수행문만 수행 | parameter를 사용, 수행문만 수행 |
    | 반환 값 존재 | parameter없이, 수행문 수행 후 결과값 반환 | parameter를 사용하여 수행문 수행 후 결과값 반환 |

### Console I/O

- input() : 콘솔창에서 문자열을 입력 받는 함수
- print() : , 사용할 경우 print 문이 연결됨

#### print formatting

1) %string
2) format 함수
3) fstring

- old-school formatting
  - 일반적으로 %-format와 str.format() 함수를 사용한다

- %-format
  - "%datatype" % (varaible) 형태로 출력 양식을 표현

- fstring
  - python 3.6 이후, PEP498에 근거한 formatting 기법
    ```python
    name = "gootea"

    age = 99

    print(f"Hello, {name}. You are {age}")
    # Hello, gootea. You are 99
    ```

## Day 8 : Conditionals and Loops

### condition (조건)

- 조건문
  - 조건에 따라 특정한 동작을 하게하는 명령어
    - 조건을 나타내는 기준
    - 실행해야 할 명령
  - 조건에 따라 명령이 실행되거나 실행되지 않음
  - if, else,elif

- 논리 키워드
  - and
  - or
  - not

- 삼항 연산자
  - 조건문을 사용하여 참일 경우와 거짓일 경우의 결과를 한줄로 표현

    ```python
    value = 12
    is_even = True if value % 2 == 0 else False
    print(is_even) # True
    ```

### loop

- 반복문
  - 정해진 동작을 반복적으로 수행하게 하는 명령문
    - 반복 시작 조건
    - 종료 조건
    - 수행 명령
  - for, while

- for 문
  - 기본적인 반복문 : 반복 범위를 지정하여 반복문 수행
  - range() 사용

  - for문의 다양한 반복 조건
    - 문자열을 한자씩 리스트로 처리 - 시퀀스형 자료형
    - 각각의 문자열 리스트로 처리
    - 간격을 두고 세기
    - 역순으로 반복문 수행

- while 문
  - 조건이만족하는 동안 반복 명령문을 수행
  - for문을 while문으로 변환 가능

  - 반복 제어
    - break : 특정 조건에서 반복 종료
    - continue : 특정 조건에서 남은 반복 명령 skip

  - else : 반복 조건이 만족하지 않을 경우 반복 종료 시 1회 수행


#### 반복문 상식

- 반복문 변수명
  - 대부분 i,j,k 사용
- 0부터 시작하는 반복문
  - 대부분 0부터 반복을 시작
  - 2진수가 0부터 시작해서 0부터 시작하는 것을 권장
- 무한 loop
  - 반복 명령이 끝나지 않는 프로그램 오류
  - CPU와 메모리 등 컴퓨터의 리소스를 과다하게 점유

## Day 9 : String and advanced function concept

### string

- 시퀀스형 자료형
- 문자형 data를 메모리에 저장
- 영문자 한 글자는 1 byte의 메모리 공간을 사용
- 1 byte 크기로 한 글자씩 메모리 공간이 할당됨

- 컴퓨터는 2진수로 데이터 저장
- 이진수 한 자릿수는 1 bit로 저장
- 1 bit = 0 또는 1
- 1 byte = 8 bit = 2^8 = 256 까지 저장 가능

- 컴퓨터는 모든 데이터를 2진수로 인식
- 2진수를 문자로 변환하는 표준 규칙을 정한다

- 프로그램 언어에서 데이터 타입
  - 각 타입 별로 메모리 공간을 할당 받은 크기가 다름
    - int : 4 byte
    - long : 무제한
    - float : 8 byte

- 문자열 특징 - indexing
  - 문자열의 각 문자는 개별 주소를 가짐
  - 이 주소를 사용해 할당된 값을 가져오는 것
  - List와 같은 형태로 데이터를 처리
  - 왼쪽에서는 0부터, 오른쪽에서는 -1부터 시작

- 문자열 특징 - slicing
  - 문자열의 주소값을 기반으로 문자열의 부분값을 반환

- 문자열 연산 및 포함여부 검사
  - 덧셈, 뺄셈 연산 가능, in 명령으로 포함여부 체크

- 문자열 함수

  | 함수명 | 기능 |
  | - | - |
  | `len(a)` | 문자열의 문자 개수를 반환 |
  | `a.upper()` | 대문자로 변환 |
  | `a.lower()` | 소문자로 변환 |
  | `a.capitalize()` | 첫 문자를 대문자로 변환 |
  | `a.count('abc')` | 문자열 `a`에 `'abc'`가 들어간 횟수 반환 |
  | `a.find('abc')`  | 문자열 `a`에 `'abc'`가 들어간 위치(오프셋) 반환 |
  | `a.rfind('abc')` | 문자열 `a`에 `'abc'`가 들어간 위치(오프셋) 반환 (뒤에서 검색) |
  | `a.strip()` | 좌우 공백을 없앰 |
  | `a.rstrip()` | 오른쪽 공백을 없앰 |
  | `a.lstrip()` | 왼쪽 공백을 없앰 |
  | `a.split()` | 공백을 기준으로 나눠 리스트로 반환 |
  | `a.split('abc')` | `'abc'`를 기준으로 나눠 리스트로 반환 |
  | `a.isdigit()` | 문자열이 숫자인지 여부 반환 |
  | `a.islower()` | 문자열이 소문자인지 여부 반환 |
  | `a.isupper()` | 문자열이 대문자인지 여부 반환 |

### function 2

- 함수에서 parameter를 전달하는 방식
  1) Call by Value
   - 함수에 인자를 넘길 때 값만 넘김
   - 함수 내에 인자 값 변경 시, 호출자에게 영향을 주지 않음
  2) Call by Reference
   - 함수에 인자를 넘길 때 메모리 주소를 넘김
   - 함수 내에 인자 값 변경 시, 호출자의 값도 변경됨
  3) Call by Object Reference
   - 파이썬은 **객체의 주소가 함수**로 전달되는 방식
   - 전달된 객체를 참조하여 변경 시 호출자에게 영향을 주나, 새로운 객체를 만들 경우 호출자에게 영향을 주지 않음

- 변수의 범위
  - 변수가 사용되는 범위
    - 지역 변수 : 함수 내에서만 사용
    - 전역 변수 : 프로그램 전체에서 사용
  - global
    - 함수 내에서 전역 변수 사용 시 global 사용

- 재귀 함수
  - 자기 자신을 호출하는 함수
  - 점화식과 같은 재귀적 수학 모형을 표현할 때 사용
  - 재귀 종료 조건 존재, 종료 조건까지 함수 호출 반복

## Day 10 : Python data structure

- python 기본 데이터 구조
  - stack, queue
  - tuple, set
  - dictionary
  - Collection 모듈

- string
  - 시퀀스 자료형으로 문자형 data를 메모리에 저장
  - 영문자 한 글자는 1byte 메모리 공간 사용

- stack
  - 나중에 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조
  - 후입 선출 (LIFO)
  - data 입력 == push == append()
  - data 출력 == pop == pop()
  - **리스트**로 구현

- queue
  - 먼저 넣은 데이터를 먼저 반환하도록 설계된 메모리 ㅣ구조
  - 선입 선출 (FIFO)
  - **리스트**로 구현
  - put == append()
  - get == pop(0)

- tuple
  - **값의 변경이 불가능한 리스트**
  - () 사용
  - 리스트의 연산, indexing, slicing 등 동일하게 사용
  - 변경되지 않은 데이터의 저장
  
- set
  - 값을 **순서없이 저장, 중복 불허**하는 자료형
  - set 객체 선언을 이용하여 객체 생성
  - 수학에서 활용하는 다양한 **집합 연산 가능**

- dictionary
  - 데이터를 저장할 때는 구분 지을 수 있는 값을함께 저장
  - 구분을 위한 데이터 고유 값을 key 라고 함
  - key 값을 활용하여 value 를 관리
  - key, value를 매칭하여 key로 value 검색
  - 다른 언어에서는 Hash Table 이라는 용어를 사용

- collections
  - list, tuple, dict에 대한 확장 자료 구조 모듈
  - 편의성, 실행 효율 등을 제공
    - deque
    - Counter
    - OrderedDict
    - defaultdict
    - namedtuple

  - deque
    - stack과 Queue를 지원
    - list에 비해 효율적인, 빠른 자료 저장 방식 지원
    - rotate, reverse 등 linked list의 특성 지원
    - 기존 list 형태의 함수 모두 지원

  - OrderedDict
    - 데이터를 입력한 순서대로 dict 반환
    - 3.6 버전 부터 입력한 순서를 보장하여 출력

  - defaultdict 
    - Dict type의 값에 기본 값을 지정, 신규값 생성시 사용

  - Counter
    - Sequence type의 data element들의 갯수를 dict 형태로 반환
    - Dict type, keyword parameter 등도 모두 처리 가능
    - set의 연산 지원

  - namedtuple
    - Tuple 형태로 **Data 구조체**를 저장
    - 저장되는 data의 variable을 사전에 지정해서 저장

## Day 11 : Pythonic code

- 파이썬 스타일의 코딩 기법
- **파이썬 특유의 문법**을 활용하여 효율적으로 코드 표현
- 더 이상 파이썬 특유는 아님, 많은 언어들이 서로의 장점을 채용
- 고급 코드를 작성 할 수록 더 많이 필요해짐

- split & join
- list comprehension
- enumerate & zip
- lambda & map & reduce
- generator
- asterisk

### Split & Join

- split
  - string type의 값을 기준값으로 나눠서 list 형태로 변환

- join
  - string 으로 구성된 list를 합쳐서 하나의 string 으로 변환

### list comprehension

- 기존 list 사용하여 간단히 다른 list를 만드는 방법
- 포괄적인 list, 포함되는 리스트라는 의미로 사용됨
- 파이썬에서 가장 많이 사용되는 기법 중 하나
- for + append 보다 속도가 빠름

### enumerate & zip

- enumerate : list의 element를 추출할 때 번호를 붙여서 추출
- zip : 두 개의 list의 값을 병렬적으로 추출

### lambda & map & reduce

- lambda
  - 함수 이름 없이, 함수처럼 쓸 수 있는 익명함수
  - 수학의 람다 대수에서 유래

- map
  - 두 개 이상의 list에도 적용 가능, if fliter 도 사용가능

- reduce
  - list에 똑같은 함수를 적용해서 통합

#### 3가지 다 간단하 코드로 다양한 기능 제공, 그러나 python3 에서 사용 권장하지는 않음

### iterable object

- Sequence형 자료형에서 데이터를 순서대로 추출하는 object
- 내부적 구현으로 __iter__와 __next__가 사용됨
- iter() 와 next() 함수로 iterable 객체를 iterator object로 상ㅇ

### generator

- iterable object를 특수한 형태로 사용해주는 함수
- element가 사용되는 시점에 값을 메모리에 반환
  - yield를 사용해 한번에 하나의 element만 반환함

- list comprehension과 유사한 형태
- generator 형태의 list 생성 == generator expression
- [] 대신 () 를 사용하여 표현

- list 타입의 데이터를 반환해주는 함수를 generator 로 제작
- 큰 데이터를 처리할 때 사용
- 파일 데이터를 처리할 때도 사용

### function passing arguments

1) keyword arguments
2) default arguments
3) variable length arguments

- 함수에 입력되는 parameter의 변수명을 사용, arguments를 넘김
- parameter의 기본 값을 사용, 입력하지 않을 경우 기본값 출력

### variable length asterisk

- 가변인자
  - **개수가 정해지지 않은 변수**를 함수의 parameter로 사용하는 법
  - keyword arguments와 함께, argument 추가가 가능
  - Asterisk(*) 기호를 사용하여 함수의 parameter를 표시
  - 입력된 값은 tuple type으로 사용 가능
  - 가변인자는 오직 한개만 맨 마지막 parameter 위치에 사용가능
  - 일반적으로 변수명을 *args 사용
  - 기존 parameter 이후에 나오는 값을 tuple로 저장

- 키워드 가변인자
  - parameter 이름을 따로 지정하지 않고 입력하는 방법
  - Asterisk 2개(**)를 사용하여 함수의 parameter를 표시
  - 입력된 값은 **dict type**으로 사용 가능
  - 가변인자는 오직 한 개만 기존 가변인자 다음에 사용

- asterisk
  - `*` 를 의미
  - 단순 곱셈, 제곱 연산, 가변 인자 활용 등 사용
  - tuple, dict 등 자료형에 있는 값 unpacking
  - 함수의 입력값, zip 등에 유용하게 사용 가능

## Day 12 : OOP (객체 지향 프로그래밍 언어)

- Object Oriented Programming
- 객체 : 실생활에서 일종의 물건
  - 속성 (Attribute) 와 행동 (Action) 을 가짐
- OOP는 이러한 객체 개념을 프로그램으로 표현
  - 속성 == 변수 (variable)
  - 행동 == 함수 (method)

- OOP 구성
  - class
  - instance (실제 구현체)

#### python naming rule

- 변수와 Class명 함수명 짓는 방식이 존재
- snake_case : 띄어쓰기 부분 == '_'
- CamelCase : 띄어쓰기 부분에 대문자 사용, Class명에 사용

- Attribute 추가
  - __init__, self 사용
  - __init__ : **객체 초기화 예약 함수**

    ```python
    class SoccerPlayer(object) :
        def __init__(self, name, position, back_number):

            self.name= name
            self.position = position
            self.back_number = back_number
    ```

- method 구현하기
  - self를 추가해야만 class 함수로 인정됨

- objects(instance) 사용하기
  - Object 이름 선언과 함께 초기값 입력

### OOP 특성

- Inheritance (상속)
  - 부모 클래스부터 속성과 Method를 물려받은 자식 클래스를 생성
- Polymorphism (다형성)
  - 같은 이름 메소드의 **내부 로직을 다르게 작성**
  - 같은 부모클래스의 상속에서 주로 발생
- Visibility (가시성)
  - 객체의 정보를 볼 수 있는 레벨을 조절
  - **누구나 객체 안에 모든 변수를 볼 필요가 없음**
    1) 객체를 사용하는 사용자가 임의로 정보 수정
    2) 필요 없는 정보에는 접근할 필요가 없음
    3) 만약 제품을 판매한다면? 소스의 보호

#### Encapsulation

- 캡슐화, 정보 은닉
- Class를 설계할 때, 클래스 간 간섭 / 정보공유의 최소화

