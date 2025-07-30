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
 