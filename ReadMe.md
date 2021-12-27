<!-- Header -->
# 모바일 게임플레이 행동 정책 학습을 위한 YOLOv3 기반 강화학습
## 정보(Information)
--- 

- 대한전기학회 논문지 게재(논문번호 : 2021-D-CA-0062) (2021.12)<br>
The paper of the Korean Institute of Electrical Engineers(KIEE) was published (Paper number: 2021-D-CA-0062) (Dec, 2021.)
<br><br>
- 대한전자공학회 추계학술대회 학부생논문경진부문 장려상 수상 (2021.11)<br>
Winning the Encouragement Award for Undergraduate Paper Competition at the Autumn Conference of the Institute of Electronics and Information Engineers(IEIE) (Nov, 2021.)
<br><br><br><br><br>

## 개요(Outline)
--- 

- 구글플레이 앱스토어 내 'Timberman'이라는 명칭의 모바일 게임에 대하여 게임이미지를 기반으로 게임플레이정책을 학습하는 심층강화학습 모델<br>
Deep reinforcement learning model that learns game play policies based on game images for mobile games named "Timberman" in the Google Play App Store.
<br><br>
- 심층강화학습 학습모델구조인 A2C(Advantage Actor-Critic) 모델을 기초로 하여 A2C 학습모델에 전이학습을 적용하여 학습시킨 T-A2C(Transfer learning based Advantage Actor-Critic) 모델, 비동기적 방법에 기초한 분산강화학습 학습모델구조인 A3C(Asynchronous Actor-Critic) 모델과의 학습능력비교<br>
Comparison of learning performance between A2C, a deep reinforcement learning model, and T-A2C model, an A2C model applied with Transfer learning, and A3C model, a Distributed deep reinforcement learning model.


<br><br><br><br><br>

## 개발 환경(Development environment)
--- 

<table>
<thead>
<tr>
<th>항목(List)</th>
<th>내용(Content)</th>
</tr>
</thead>
<tbody>
<tr>
    <td>하드웨어 개발 환경(HW-Environment)</td>
    <td>
    <img src="https://img.shields.io/badge/CPU-i5_6500-0078D6?
                style=plastic
                &logo=Windows
                &logoColor=white"/>
    <img src="https://img.shields.io/badge/GPU-GTX_1060-44A833?
                style=plastic
                &logo=Anaconda
                &logoColor=white"/>
    </td>
</tr>
<tr>
    <td>소프트웨어 개발 환경(SW-Environment)</td>
    <td>
    <img src="https://img.shields.io/badge/OS-Windows-0078D6?
                style=plastic
                &logo=Windows
                &logoColor=white"/>
    <img src="https://img.shields.io/badge/ENV-Anaconda-44A833?
                style=plastic
                &logo=Anaconda
                &logoColor=white"/>
    </td>
</tr>
<tr>
    <td>개발 언어(Language)</td>
    <td>
    <img src="https://img.shields.io/badge/LANG-Python-3776AB?
                style=plastic
                &logo=Python
                &logoColor=white"/>
    </td>
</tr>
<tr>
    <td>라이브러리(Library)</td>
    <td>
    <img src="https://img.shields.io/badge/DL-PyTorch-EE4C2C?
                style=plastic
                &logo=PyTorch
                &logoColor=white"/> 
    <img src="https://img.shields.io/badge/DL-NumPy-013243?
                style=plastic
                &logo=NumPy
                &logoColor=white"/>
    <img src="https://img.shields.io/badge/IMG_PROCESS-OpenCV-5C3EE8?
                style=plastic
                &logo=OpenCV
                &logoColor=white"/>
    </td>
</tr>
<tr>
    <td>보조(외부) 프로그램(External program)</td>
    <td>
    <img src="https://img.shields.io/badge/Web_Cam-ManyCam-00CEC8?
                style=plastic
                &logo=ManyCam
                &logoColor=white"/>
    </td>
</tr>
</tbody>
</table>

<br><br><br><br><br>

## 시연영상(Demonstration video)
--- 

![시연영상_1](.\ReadMe\설계변환후_1.gif "시연영상_1")
![시연영상_2](.\ReadMe\설계변환후_2.gif "시연영상_2")

<br><br><br><br><br>

## 구조(Structure)
--- 

### 학습모델 구조도(학습모델의 반복학습 절차)
### Structure diagram of learning model (repeated learning procedure of learning model)
![A2C,T-A2C모델구조](.\ReadMe\A2C,T-A2C모델구조.jpg "A2C,T-A2C모델구조")

<ol>
    1. 상태추출 : S_t >> 현재상태 추출<br>
    2. 행동추출 및 실행 : A_t >> 선택 가능한 행동 중 택 1<br>
    3. 보상, 다음상태 추출
    <ol>
        3-1. 보상획득 : R_t+1 >> 행동에 대한 결과로 보상 획득 >> 학습 <br>
        3-2. 다음상태추출 : S_t+1 >> 행동에 대한 결과로 나타난 다음상태 추출<br>
    </ol>
</ol>

<br><br><br><br><br>

### 보상함수 구조도
### Structure diagram of the reward function
![보상함수](.\ReadMe\보상함수.gif "보상함수")
<ol>
    1. 기본 보상(Basic)<br>
    <ul>
        행동에 따라 생존 시 → +0.5 수여<br>
        행동에 따라 사망 시 → -1 수여<br>
    </ul><br>
    2. 인센티브 보상(Incentive)<br>
    <ul>
        위험상태(장애물 인접상태)에서 행동에 따라 생존 시 → +0.5 수여<br>
    </ul>
</ol>


<br><br><br><br><br>

## 구현(Implementation)
---

### 프로그램 흐름도(Flow chart)

![프로그램흐름도](.\ReadMe\프로그램흐름도.jpg "프로그램흐름도")

<ol>
    1. 상태 추출<br>
    <ol>
        1-1. 현재상태 데이터(게임이미지 : Webcam Real-Time Image Stream) >> 분류 신경망(실시간 객체탐지기 : YOLOv3)<br>
        1-2. 현재상태 전처리 데이터 >> 심층강화학습 신경망(A2C/T-A2C/A3C) 입력<br>
    </ol><br>
    2. 행동 추출(심층강화학습 신경망 출력) 및 실행(게임 내 입력)<br><br>
    3. 보상, 다음상태 데이터<br>
    <ol>
        3-1. 보상 데이터(상태 입력 : 보상함수 출력) >> 심층강화학습 신경망 업데이트<br>
        3-2. 다음상태 데이터 >> 현재상태 데이터로서 처리(반복)<br>
    </ol><br>
</ol>

<br><br><br><br><br>

## 특징점(Development feature)
---

### 상태도메인 축소(State domain reduction)
![상태3진화_1](.\ReadMe\상태3진화_1.jpg "상태3진화_1")
![상태3진화_2](.\ReadMe\상태3진화_2.jpg "상태3진화_2")

- 상태 표현 3진화에 따라 상태도메인 축소<br><br>
- 변경 전(픽셀 수 = 900x1600 = 1,440,000개) → 변경 후(층계 별 장애물 정보 = 6개)

<br><br><br>

### 전이학습 적용(Transfer learning application)
![전이학습](.\ReadMe\전이학습.jpg "전이학습")

- 본 게임 환경과 유사한 코드 레벨 가상 환경 모듈에 대해 사전 학습 1000회 수행<br><br>
- 사전 학습을 수행함으로써 초기 유리한 가중치 설정 및 궁극적인 학습 성능 향상 도모

<br><br><br>

### 분산강화학습 적용(Distributed reinforcement learning application)
![A3C모델구조](.\ReadMe\A3C모델구조.jpg "A3C모델구조")

- 다개체 기반 학습모델 알고리즘(A3C)를 활용한 분산강화학습을 통해 궁극적인 학습 성능 향상 도모<br><br>
- 멀티 에이전트 모델의 반복학습 절차
<ol>
    1. (워커 에이전트)샘플 추출 : T_n >> 현재(n차) 에피소드 데이터(Trajectory) 추출 >> 글로벌 네트워크로 전송<br>
    2. (글로벌 네트워크)비동기적 학습 : 다개체의 워커에이전트로부터 비동기적으로 추출된 샘플 학습 >> 학습 가중치 정보 역전송<br>
    3. (워커 에이전트)학습 : 글로벌 네트워크로부터 받은 학습 가중치 정보를 기반으로 학습
</ol>


<br><br><br><br><br>

## 학습성능 및 실험결과(Learning performance and Experimental results)
---

![T-A2C성능](.\ReadMe\T-A2C성능.jpg "T-A2C성능")
### A2C / T-A2C 모델 간 성능 비교(Learning performance comparison between A2C and T-A2C)
<ol>
    1. 액터 신경망
    <ol>
    가. 초기 : +2.399 → -0.6035(79.9%)<br>
    나. 최종 : -0.8986 → -1.379(34.8%)<br>
    </ol><br>
    2. 크리틱 신경망
    <ol>
    가. 초기 : +14.51 → +1.437(90.1%)<br>
    나. 최종 : +0.9469 → +0.7643(19.3%)<br>
    </ol><br>
    3. 누적 보상
    <ol>
    가. 최종 : +0.8096 → +10.27(21.2%)<br>
    </ol>
</ol>

<br><br><br>

![A3C성능](.\ReadMe\A3C성능.jpg "A3C성능")
### A2C / A3C 모델 간 성능 비교(Learning performance comparison between A2C and A3C)
<ol>
    1. 액터 신경망
    <ol>
    가. 최종 : -0.8986 → -3.674(75.5%)<br>
    </ol><br>
    2. 크리틱 신경망
    <ol>
    가. 최종 : +0.9469 → +0.9329(1.5%)<br>
    </ol><br>
    3. 누적 보상
    <ol>
    가. 최종 : +0.8096 → +12.75(36.5%)<br>
    </ol>
</ol>

<br><br><br>

![실험결과](.\ReadMe\실험결과.jpg "실험결과")
### 모델 별 10회 실험결과(Result of 10 experiments per model)
- A3C 모델의 10회 실험 간 취득점수 기댓값 : 174점<br><br>
- T-A2C 모델의 10회 실험 간 취득점수 기댓값 : 173.5점<br><br>
- A2C 모델의 10회 실험 간 취득점수 기댓값 : 146점<br>
** 일부 시도에서 학습 성능 불안정성에 따라 낮은 점수를 기록

<br><br><br><br><br>
