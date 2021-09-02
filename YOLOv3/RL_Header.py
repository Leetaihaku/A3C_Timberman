import Main_Header as Main
import torch
import torch.nn.functional as F
import torchsummary as ts
import keyboard
import numpy as np
import os.path as op
import time

from torch import nn
from torch import optim
from collections import namedtuple
from collections import deque

# TD(0), TD(n) 모드 선택변수
MODE = 1
# 상태 차원, 가치 차원
STATE_DIM = 6
VALUE_DIM = 1
# 행동, 행동 차원
ACTION_OPTION = ['a', 'd']
ACTION_DIM = 2
# 학습률
EPSILON_LOWER_LIMIT = 1e-3
# 할인률
GAMMA = 0.9
GAMMA_LIST = [
    [[GAMMA**0]],
    [[GAMMA**0], [GAMMA**1]],
    [[GAMMA**0], [GAMMA**1], [GAMMA**2]],
    [[GAMMA**0], [GAMMA**1], [GAMMA**2], [GAMMA**3]],
    [[GAMMA**0], [GAMMA**1], [GAMMA**2], [GAMMA**3], [GAMMA**4]],
    [[GAMMA**0], [GAMMA**1], [GAMMA**2], [GAMMA**3], [GAMMA**4], [GAMMA**5]],
    [[GAMMA**0], [GAMMA**1], [GAMMA**2], [GAMMA**3], [GAMMA**4], [GAMMA**5], [GAMMA**6]],
    [[GAMMA**0], [GAMMA**1], [GAMMA**2], [GAMMA**3], [GAMMA**4], [GAMMA**5], [GAMMA**6], [GAMMA**7]],
    [[GAMMA**0], [GAMMA**1], [GAMMA**2], [GAMMA**3], [GAMMA**4], [GAMMA**5], [GAMMA**6], [GAMMA**7], [GAMMA**8]]
]
# 배치형식
BATCH = namedtuple('BATCH', ('state', 'action', 'reward', 'next_state'))
# 배치사이즈
BATCH_SIZE = 4
# 자리표현 딕셔너리
STATE_DIC = {'11': 0b100000000000, '12': 0b010000000000, '21': 0b001000000000, '22': 0b000100000000,
             '31': 0b000010000000, '32': 0b000001000000, '41': 0b000000100000, '42': 0b000000010000,
             '51': 0b000000001000, '52': 0b000000000100, '61': 0b000000000010, '62': 0b000000000001,
             '00': 0b000000000000}
# 5상태 생성 보조용 딕셔너리
FUTURE5_ASSISTANT = ['00', '10', '10', '10', '10']
FUTURE_ASSISTANT = '10'

# 플레이 안정화 지연
TRAIN_ACTION_DELAY = 0.2
TRAIN_DETECT_DELAY = 0.1
TEST_ACTION_DELAY = 0.175
TEST_DETECT_DELAY = 0.1



def Print_all(state, action, reward, next_state, v_value, next_v_value, q_value, advantage):
    """상태 출력(디버깅 용도)"""
    print('############################################################################')
    print('')
    print('State : \t\t', state)
    print('Action : \t\t', action)
    print('Next_state : \t', next_state)
    print('Reward : \t\t', reward)
    print('V_value : \t\t', v_value)
    print('Q_value : \t\t', q_value)
    print('Next_V_value : \t', next_v_value)
    print('Advantage : \t', advantage)
    print('')
    print('############################################################################')
    return


def Actor_network(NODES):
    """액터-신경망"""
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(STATE_DIM, NODES))
    model.add_module('relu', nn.ReLU())
    model.add_module('drop', nn.Dropout(p=0.5))
    model.add_module('fc2', nn.Linear(NODES, ACTION_DIM))
    model.add_module('log_softmax', nn.LogSoftmax(dim=0))
    return model


def Critic_network(NODES):
    """크리틱-신경망"""
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(STATE_DIM, NODES))
    model.add_module('relu', nn.ReLU())
    model.add_module('drop', nn.Dropout(p=0.5))
    model.add_module('fc2', nn.Linear(NODES, VALUE_DIM))
    return model




class Agent:
    """에이전트"""
    def __init__(self, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size):
        self.Actor = Actor_network(node).cuda()
        self.Critic = Critic_network(node).cuda()
        self.Optimizer1 = optim.Adam(self.Actor.parameters(), lr=learning_rate)
        self.Optimizer2 = optim.Adam(self.Critic.parameters(), lr=learning_rate)
        self.Actor_loss = torch.zeros(1, device='cuda')
        self.Actor_loss_stack = torch.zeros(1, device='cuda')
        self.Critic_loss = torch.zeros(1, device='cuda')
        self.Critic_loss_stack = torch.zeros(1, device='cuda')
        self.Reward_stack = torch.zeros(1, device='cuda')
        self.Step_stack = 0
        self.Epsilon = epsilon
        self.Epsilon_discount = epsilon_discount
        self.Epsilon_lower_limit = EPSILON_LOWER_LIMIT
        self.Batch = deque()
        self.index = 0
        self.Step_mode = step_mode
        self.Batch_size = batch_size
        # 모델 요약
        # ts.summary(self.Actor, (1, STATE_DIM), device='cuda')
        # ts.summary(self.Critic, (1, STATE_DIM), device='cuda')


    def Start(self, mode='train'):
        """게임 시작"""
        # 가중치 불러오기
        if self.Pre_trained_model_check(mode) and mode == 'train':
            self.Actor.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBA))
            self.Critic.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBC))
        elif self.Pre_trained_model_check(mode) and mode == 'test':
            self.Actor.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBA_TEST))
            self.Critic.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBC_TEST))
        # 추론모드 설정
        self.Actor.eval()
        self.Critic.eval()
        # 게임 시작
        if mode == 'train':
            keyboard.press_and_release('s')
        return


    def Pre_trained_model_check(self, mode):
        """사전 훈련된 모델 검색"""
        if mode == 'train':
            return True if op.isfile(Main.MODEL_PATH+Main.TBA) \
                           and op.isfile(Main.MODEL_PATH+Main.TBC) else False
        elif mode == 'test':
            return True if op.isfile(Main.MODEL_PATH+Main.TBA_TEST) \
                           and op.isfile(Main.MODEL_PATH+Main.TBC_TEST) else False


    def Action(self, state, mode='train'):
        """에이전트 행동 추출"""
        self.Epsilon = self.Epsilon - self.Epsilon_discount if self.Epsilon > self.Epsilon_lower_limit else self.Epsilon_lower_limit
        action = self.Actor(self.State_to_network_input(state))
        action = action.argmin().item() if self.Epsilon > np.random.uniform(0, 1) else action.argmax().item()
        print(f'action : {action}')
        if mode == 'train':
            keyboard.press_and_release(ACTION_OPTION[action])
        return action


    def Save_batch(self, state, action, reward, next_state):
        """큐(Queue)배치 저장"""
        self.Batch.append(None)
        self.Batch[self.index] = BATCH(state, action, reward, next_state)
        self.index += 1 if self.index < self.Batch_size-1 else 0


    def State_to_network_input(self, origin_state):
        """탐지상태 → 네트워크 입력상태 변환"""
        if int(origin_state[1]) == 0:
            return torch.tensor([0., 0., 0., 0., 0., 0.], device='cuda')
        else:
            converted_branch = 0b000000000000
            branch = str(int(origin_state[0])) if str(int(origin_state[0])) != '0' else '00'
            for i in range(0, len(branch), 2):
                converted_branch |= STATE_DIC[branch[i:i+2]]
            converted_branch = format(converted_branch, 'b').zfill(12)
            network_input = []
            for i, j in zip(converted_branch[::2], converted_branch[1::2]):
                network_input.append(0. if int(i + j, 2) == 0 else -1. if int(i+j, 2) == 2 else 1.)
        return torch.as_tensor(network_input, device='cuda')


    def Variable_ready_for_TD_0(self, state, reward, next_state):
        """어드밴티지 및 행동가치함수 계산-TD(0)"""
        v_value = self.Critic(state)
        next_v_value = self.Critic(next_state)
        q_value = reward + GAMMA * next_v_value
        advantage = q_value - v_value
        return v_value, advantage, q_value


    def Variable_ready_for_TD_N(self):
        """신경망 업데이트 재료변수 계산-TD(N)"""
        # 그래디언트 중복방지용 연산대상 인스턴스 생성
        state = self.Batch[0][0]
        v_value = self.Critic(self.Batch[0][0])
        reward = self.Batch[0][2]
        next_v_value = self.Critic(self.Batch[-1][3])

        # Step 별 update 필요변수 준비
        partial_g = torch.zeros(1, device='cuda')
        # 부분 반환값 계산1 (보상 GAMMA 감가)
        for i in range(len(self.Batch)):
            self.Batch[i][2] = torch.mul(self.Batch[i][2], GAMMA**i)
            partial_g = torch.add(partial_g, self.Batch[i][2])
        # 부분 반환값 계산2 (N-step 상태가치함수 GAMMA 감가)
        discounted_next_v_value = torch.mul(next_v_value, GAMMA**(len(self.Batch))).detach()

        # 행동가치함수 계산
        q_value = torch.add(partial_g, discounted_next_v_value)
        # Advantage 계산
        advantage = torch.sub(q_value, v_value).detach()
        return state, v_value, reward, advantage, q_value


    def Variable_ready_for_TD_N_Parallel(self):
        """신경망 업데이트 재료변수 계산-TD(N) 병렬계산"""
        # 그래디언트 중복방지용 연산대상 인스턴스 생성 from 배치
        batch = BATCH(*zip(*self.Batch))
        state = batch.state[0]
        v_value = self.Critic(self.State_to_network_input(state))
        action = batch.action[0]
        reward_serial = torch.mul(torch.stack(batch.reward), torch.as_tensor(GAMMA_LIST[len(self.Batch)-1], device='cuda'))
        reward = reward_serial[0]
        next_state = batch.next_state[-1]
        next_v_value = self.Critic(self.State_to_network_input(next_state))

        # 반환값 계산(보상 감가 & 상태_t_n 감가)
        partial_g = torch.sum(reward_serial, dim=0)
        discounted_next_v_value = torch.mul(next_v_value, GAMMA*GAMMA_LIST[len(self.Batch)-1][-1][0])

        # 행동가치함수 계산
        q_value = torch.add(partial_g, discounted_next_v_value)
        # Advantage 계산
        advantage = torch.sub(q_value, v_value).detach()
        return state, v_value, action, reward, advantage, q_value


    def Update_by_TD(self, state, v_value, action, reward, advantage, q_value):
        """신경망 업데이트(TD(0))"""
        # 액터-신경망 훈련
        self.Actor.train()
        self.Optimizer1.zero_grad()
        self.Actor_loss = - self.Actor(self.State_to_network_input(state))[action] * advantage
        self.Actor_loss.backward(retain_graph=True)
        self.Optimizer1.step()
        self.Actor.eval()

        # 크리틱-신경망 훈련
        self.Critic.train()
        self.Optimizer2.zero_grad()
        self.Critic_loss = F.mse_loss(torch.squeeze(v_value), torch.squeeze(q_value))
        self.Critic_loss.backward()
        self.Optimizer2.step()
        self.Critic.eval()

        # export 용 에피소드 데이터 축적
        self.Actor_loss_stack += self.Actor_loss
        self.Critic_loss_stack += self.Critic_loss
        self.Reward_stack += reward
        return


    def Action6_executer(self, future_arr):
        """테스트 :: 5단계 행동시퀀스 생성"""
        tensor5 = torch.tensor([], device='cuda')
        for i in range(len(future_arr)):
            tensor5 = torch.cat((tensor5, self.State_to_network_input(future_arr[i])), dim=0)
        action6 = self.Actor(torch.reshape(tensor5, (5, 6))).max(1)[1]
        for i in range(5):
            keyboard.press_and_release(ACTION_OPTION[action6[i].item()])
            time.sleep(TRAIN_ACTION_DELAY)
        return


    def Step5_training(self, state):
        """5-step 훈련 :: 1-step 미래상태 생성"""
        for step in range(5):
            # 상태에 따른 정책행동 실행 & 안정화 지연
            print(f'state : {state}')
            mini_action = self.Action(state)
            time.sleep(TRAIN_ACTION_DELAY)
            # 행동에 따른 다음상태 케이스 분류
            # 위험상태에서의 나쁜 행동
            if (str(int(state[0].item()))[:2] in ['51', '61'] and mini_action == 0) or (str(int(state[0].item())))[:2] in ['52', '62'] and mini_action == 1:
                done = True
                reward = torch.tensor([-1.], device='cuda')
                next_state = torch.tensor([0., 0., 0., 0., 1.], device='cuda')
            # 좋은(일반) 행동
            else:
                done = False
                reward = torch.tensor([1. if str(int(state[0].item()))[:2] in ['51', '52', '61', '62'] else 0.5], device='cuda')
                next_state = ''
                for half_branch_str in range(len(str(int(state[0].item()))) >> 1):
                    next_state += FUTURE_ASSISTANT
                next_state = str((int(next_state) if next_state != '' else 0) + (int(state[0].item()) if state != '' else 0))
                # 인덱스 초과 상태에 대한 전처리
                for half_future_branch_str in range(len(next_state) >> 1):
                    next_state = next_state[2:] if int(next_state[0]) > 6 else next_state
                next_state = torch.tensor([float(next_state) if next_state != '' else 0., 61. if mini_action == 0 else 62., 0., 0., 0.], device='cuda')
            # TD(N)
            if self.Step_mode:
                # 배치 저장
                self.Save_batch(state.detach(), mini_action, reward, next_state.detach())
                # 종료 시, 잔여 업데이트 진행
                if done:
                    for batch_idx in range(len(self.Batch)):
                        state, v_value, mini_action, reward, advantage, q_value = self.Variable_ready_for_TD_N_Parallel()
                        self.Update_by_TD(state, v_value, mini_action, reward, advantage, q_value)
                        self.Batch.popleft()
                    break
                # 진행 시, 배치사이즈 한도 내에서 업데이트 진행
                elif len(self.Batch) == self.Batch_size:
                    state, v_value, mini_action, reward, advantage, q_value = self.Variable_ready_for_TD_N_Parallel()
                    self.Update_by_TD(state, v_value, mini_action, reward, advantage, q_value)
                    self.Batch.popleft()
            # TD(0)
            else:
                v_value, advantage, q_value = self.Variable_ready_for_TD_0(state, reward, next_state)
                self.Update_by_TD(state, v_value, mini_action, reward, advantage, q_value)
            # 손실 및 보상 평균연산용 분모변수, 다음상태 전환
            self.Step_stack += 1
            state = next_state
        return state, done


    def Step5_testing(self, state):
        for step in range(5):
            # 상태에 따른 정책행동 실행 & 안정화 지연
            mini_action = self.Actor(self.State_to_network_input(state)).max(0)[1].item()
            keyboard.press_and_release(ACTION_OPTION[mini_action])
            time.sleep(TEST_ACTION_DELAY)
            # 행동에 따른 다음상태 케이스 분류
            # 위험상태에서의 나쁜 행동
            if (str(int(state[0].item()))[:2] in ['51', '61'] and mini_action == 0) or (str(int(state[0].item())))[:2] in ['52', '62'] and mini_action == 1:
                done = True
                next_state = torch.tensor([0., 0., 0., 0., 1.], device='cuda')
            # 좋은(일반) 행동
            else:
                done = False
                next_state = ''
                for half_branch_str in range(len(str(int(state[0].item()))) >> 1):
                    next_state += FUTURE_ASSISTANT
                next_state = str((int(next_state) if next_state != '' else 0) + (int(state[0].item()) if state != '' else 0))
                # 인덱스 초과 상태에 대한 전처리
                for half_future_branch_str in range(len(next_state) >> 1):
                    next_state = next_state[2:] if int(next_state[0]) > 6 else next_state
                next_state = torch.tensor([float(next_state) if next_state != '' else 0., 61. if mini_action == 0 else 62., 0., 0., 0.], device='cuda')
            # 다음상태 전환
            self.Step_stack += 1
            state = next_state
        return state, done


class Environment:
    def Step(self, extracted_arr):
        """탐지화면 -> 상태식(Domain) 생성 // x-axis :: 60 ++ 50, y-axis :: 0 ++ 320(1920x1060 기준)"""
        branch = ''  # 나뭇가지 상태 -> 신경망 입력 형변환
        player = ''  # 나무꾼 상태 -> 신경망 입력 형변환
        revive_y = '0'  # 이어하기_Y 상태 -> 신경망 입력 형변환
        revive_n = '0'  # 이어하기_N 상태 -> 신경망 입력 형변환
        episode_state = '0'
        status = []  # 상태 임시저장 리스트

        # 이미지 추출 Raw 데이터 분해
        for data in extracted_arr:
            col_offset = data[1][0] // 320 + 1  # [y] +1 -> 상태 혼동 방지 bias
            row_offset = (data[1][1] - 60) // 50 + 1  # [x] -60 -> 모니터링 화면과 YOLO모델 픽셀 차이 상쇄 // +1 -> 상태혼동방지
            status.append([data[0], row_offset, col_offset])

        # 상태값 문자열 정리
        for i in range(len(status)):  # 신경망 입력 준비
            if status[i][0] == 'Branch':
                branch += str(status[i][1]) + str(status[i][2])
            elif status[i][0] == 'Player':
                player += str(status[i][1]) + str(status[i][2])
            elif status[i][0] == 'Revive_Y':
                revive_y = '1'
            elif status[i][0] == 'Revive_N':
                revive_n = '1'
            elif status[i][0] == 'Episode_Start':
                episode_state = '1'
            else:
                print('미확인 객체발생')
                exit()

        # 널 값 점검 조건부 -> 만일의 널 값 대비
        branch = str(0) if branch == '' else branch
        player = str(0) if player == '' else player
        revive_y = str(0) if revive_y == '' else revive_y
        revive_n = str(0) if revive_n == '' else revive_n

        # 나뭇가지 데이터 정제(동일상태 상이인식 방지 => 근->원)
        refined_branch = []
        for i in range(len(branch) // 2):
            refined_branch.append(int(branch[2 * i:2 * i + 2]))
        refined_branch = sorted(refined_branch, reverse=True)
        refined_branch = str(0) if refined_branch == [] else ''.join(map(str, refined_branch))

        # 다음상태 반환
        return torch.tensor([float(refined_branch), float(player),
                             float(revive_y), float(revive_n), float(episode_state)], device='cuda')


    def Reward(self, state, done):
        """보상 수여"""
        reward = torch.tensor([-1.], device='cuda') if done else torch.tensor([0.5], device='cuda')
        incentive = torch.tensor([0.5], device='cuda') if not done and str(int(state.tolist()[0]))[0] in ['5', '6'] else torch.tensor([0.], device='cuda')
        return torch.add(reward, incentive)


    def Init_state_check(self, state, next_state):
        """초기상태 확인함수"""
        state_list = state.tolist()
        return False if state_list[1] != 0. and state_list[4] != 1. and torch.equal(state, next_state) else True


    def State_diff_check(self, state, next_state, action):
        """상태변화 확인"""
        # 연산용 리스트화
        state_tolist = state.tolist()
        next_state_tolist = next_state.tolist()

        # 연산용 데이터 정제
        s_branch = str(int(state_tolist[0]))
        s_next_branch = str(int(next_state_tolist[0]))
        i_next_player = int(next_state_tolist[1])

        # 플레이어 체크(행동 → 불 변수로 취급하여 곱한 결과)
        player_ok = True if (action == 0 and i_next_player in [61, 0]) or \
                            (action == 1 and i_next_player in [62, 0]) else False

        # 나뭇가지 체크(생존여부에 따른 케이스 분류)
        branch_ok = True if self.Stochastic_check(s_branch, s_next_branch) or \
                            torch.equal(next_state, torch.tensor([0., 0., 0., 0., 1.], device='cuda')) or \
                            torch.equal(next_state, torch.tensor([0., 0., 1., 1., 0.], device='cuda')) else False
        return player_ok and branch_ok


    def Stochastic_check(self, s_branch, s_next_branch):
        """비-종료 다음상태 정답제시"""
        # 예외처리 : 두 상태 다 아무것도 없는 상태 있음
        if s_branch in ['0', '61', '62'] and s_next_branch in ['0', '11', '12']:
            return True

        # 일반처리 진행 ↓
        # 차이 변수
        correct = ''

        # 비교구간 잡기 & 실제차이 계산
        if s_branch[0] == '6':
            comp_len = len(s_branch)-2
            diff = int(s_next_branch[:comp_len]) - int(s_branch[2:])
        else:
            comp_len = len(s_branch)
            diff = int(s_next_branch[:len(s_branch)]) - int(s_branch)

        # 비교구간 정답표 생성
        for i in range(comp_len >> 1):
            correct += '10'

        # 실제차이 & 정답차이 비교
        return True if diff == int(correct) else False


    def Future6_generator(self, state):
        """테스트 :: 5단계 변환상태시퀀스 생성"""
        # 5단계 상태시퀀스
        future_state_arr = []

        # 5단계 미래 생성
        for i in range(5):
            future_state = ''
            for j in range(len(state) >> 1):
                future_state += FUTURE5_ASSISTANT[i]
            future_state = str((int(future_state) if future_state != '' else 0) + (int(state) if state != '' else 0))
            # 인덱스 초과 상태에 대한 전처리
            for k in range(len(future_state) >> 1):
                future_state = future_state[2:] if int(future_state[0]) > 6 else future_state
            future_state_arr.append(torch.tensor([float(future_state) if future_state not in ['', '0'] else 0., 61., 0., 0., 0.], device='cuda'))
            state = future_state
        return future_state_arr
