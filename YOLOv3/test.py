import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import RL_Header as RL
import subprocess
import Main_Header as Main
import numpy as np
import os.path as op
import Transfer_learning as TL
import argparse
import pyautogui
import keyboard
from collections import deque


# print(torch.tensor([3211., 61., 0., 0., 0.], device='cuda'))
# print(torch.tensor([0., 0., 1., 1., 0.], device='cuda'))
# print(torch.tensor([0., 0., 0., 0., 1.], device='cuda'))



def Actor_network():
    """액터-신경망"""
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(12, 32))
    model.add_module('relu', nn.ReLU())
    model.add_module('drop', nn.Dropout(p=0.5))
    model.add_module('fc2', nn.Linear(32, 2))
    model.add_module('log_softmax', nn.LogSoftmax(dim=0))
    return model

net = Actor_network().cuda()
print(net(torch.tensor([1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0.], device='cuda')))


# print(agent.Actor(torch.tensor([-1., -1., 0., 1., 1.], device='cuda')))
# print(agent.Actor(torch.tensor([-1., -1., 0., 1., 1.], device='cuda')))

#
# state = torch.tensor([3112., 61., 0., 0., 0.], device='cuda')
# action = 0
# reward = torch.tensor([1.], device='cuda')
# next_state = torch.tensor([4122., 61., 0., 0., 0.], device='cuda')
# agent.Save_batch(state.detach(), action, reward, next_state.detach())
#
# state = torch.tensor([4122., 61., 0., 0., 0.], device='cuda')
# action = 0
# reward = torch.tensor([1.], device='cuda')
# next_state = torch.tensor([5132., 61., 0., 0., 0.], device='cuda')
# agent.Save_batch(state.detach(), action, reward, next_state.detach())
#
# state = torch.tensor([5132., 61., 0., 0., 0.], device='cuda')
# action = 0
# reward = torch.tensor([1.], device='cuda')
# next_state = torch.tensor([6142., 61., 0., 0., 0.], device='cuda')
# agent.Save_batch(state.detach(), action, reward, next_state.detach())
#
# state = torch.tensor([6142., 61., 0., 0., 0.], device='cuda')
# action = 0
# reward = torch.tensor([1.], device='cuda')
# next_state = torch.tensor([52., 61., 0., 0., 0.], device='cuda')
# agent.Save_batch(state.detach(), action, reward, next_state.detach())
#
# batch = RL.BATCH(*zip(*agent.Batch))
# state = agent.State_to_network_input(batch.next_state[0])
# print(state)
# state = agent.State_to_network_input(batch.next_state[1])
# print(state)
# state = agent.State_to_network_input(batch.next_state[2])
# print(state)
# state = agent.State_to_network_input(batch.next_state[3])
# print(state)

#print(agent.Action6_executer(env.Future6_generator('3111')))aadaad
























# state = torch.tensor([5221., 61., 0., 0., 0.], device='cuda')
# action = 0
# reward = torch.tensor([1.], device='cuda')
# next_state = torch.tensor([6231., 61., 0., 0., 0.], device='cuda')
# agent.Batch.append([state, action, reward, next_state])
#
# state, v_v, reward, advantage, q_value = agent.Variable_ready_for_TD_N()
# print(f'advantage : {advantage}')
# agent.Actor_loss = agent.Actor(state)
# print(f'actor_output : {agent.Actor_loss}')
# agent.Actor_loss = -agent.Actor_loss[action]
# print(f'-actor_loss : {agent.Actor_loss}')
# agent.Actor_loss = agent.Actor_loss * advantage
# print(f'final : {agent.Actor_loss}')


#
#
#
# print(env.Reward(state, True))


# if len(agent.Batch) == 2:
#     state, v_v, reward, advantage, q_v = agent.Variable_ready_for_TD_NN()
#     agent.Update_by_TD(state, v_v, action, reward, advantage, q_v)
#     agent.Batch.popleft()




# keyboard.press_and_release('esc')
# time.sleep(1)
# pyautogui.moveTo(x=1665, y=995)
# pyautogui.click()
# pyautogui.moveTo(x=1020, y=90)
# pyautogui.click()
# time.sleep(1)
# keyboard.press_and_release('t')
# keyboard.press_and_release('F11')

# parser = argparse.ArgumentParser()
# parser.add_argument('--step_mode', type=int, default=1, help='True : n-step, False : 0-step')
# parser.add_argument('--batch_size', type=int, default=8, help='N-step batch_size')
# obj = parser.parse_args()
# print(bool(obj.step_mode))
# print(obj.batch_size)
# queue = deque()
# for i in range(3):
#     queue.append([torch.tensor(i, device='cuda'), True])
# print(queue)
# print(queue.pop())
# print(queue)
#
# print(queue.popleft())
# print(queue)
# print(queue[-1][-1])
# print(len(queue))





#
# agent = RL.Agent(0.999, 0.001, 0.001, 16)
# env = RL.Environment()
#
# # 종료여부에 따른 Advantage, Q_value 계산 및 업데이트
# state, done = TL.init_state_generator()
# while True:
#     action = agent.Action(state, 'test')
#     next_state, done = TL.state_generator(state, action)
#     reward = env.Reward(state, done)
#     agent.Batch.append([state.detach(), reward, next_state.detach(), done])
#     if done:
#         for i in range(len(agent.Batch)):
#             State, V_value, Reward, Advantage, Q_value = agent.Variable_ready_for_TD_N()
#             agent.Update_by_TD(State, V_value, Reward, Advantage, Q_value)
#             # 완료된 배치 대기열 큐 내 상태 제거
#             agent.Batch.popleft()
#     # 배치크기 충족 시, Advantage, Q_value 계산 및 업데이트
#     elif len(agent.Batch) == RL.BATCH_SIZE:
#         # Advantage, Q_value 계산 및 업데이트
#         State, V_value, Reward, Advantage, Q_value = agent.Variable_ready_for_TD_N()
#         agent.Update_by_TD(State, V_value, Reward, Advantage, Q_value)
#         # 완료된 배치 대기열 큐 내 상태 제거
#         agent.Batch.popleft()
#     if done:
#         break
#     else:
#         state = next_state
#
#


# init, done = TL.init_state_generator()
# while True:
#     process = 0
#     for i in range(100):
#         print(f'\n\n$$$$$$$$$$$$$$$$[ {i} ]번째 step$$$$$$$$$$$$$$$$')
#         action = random.choice([0, 1])
#         ns, done = TL.state_generator(init, action)
#         print(f'state : {init}')
#         print(f'actio : {action}')
#         print(f'nexte : {ns}')
#         if done:
#             break
#         else:
#             init = ns
#             process += 1
#     if process >= 30:
#         break


# a = TL.state_generator(torch.tensor([11., 61., 0., 0., 0.], device='cuda'), 0)
# print(int(a[0][0]))
#
# print('test'+random.choice(['11', '12']))







# 최근 학습 하이퍼파라미터 저장
# learning_info = open(Main.MODEL_PATH+Main.INFO, 'r')
# total = learning_info.readlines()
# total = total[0]+total[1]+'test\n'
# learning_info.close()
#
# learning_info = open(Main.MODEL_PATH+Main.INFO, 'w')
# learning_info.write(total)
# learning_info.close()














# state = torch.tensor([4122., 61., 0., 0., 0.], device='cuda')
# action = 1
# alive = torch.tensor([513211., 62., 0., 0., 0.], device='cuda')
# alive_alive = torch.tensor([624221., 62., 0., 0., 0.], device='cuda')
# dead_1 = torch.tensor([0., 0., 0., 0., 1.], device='cuda')
# dead_2 = torch.tensor([0., 0., 1., 1., 0.], device='cuda')
#
# env = RL.Environment()
# print(env.State_diff_check(state, alive, action))
#
# test = 'abcdef'
# print(test[:4])













# print('###abc###')
# print('###abc###')
# print('###abc###')
#
# subprocess.run([f'cd {Main.ADRESS_Activation}',
#                 'activate ai',
#                 f'cd {Main.MODEL_PATH}',
#                 'python detect.py --source 0'], shell=True)


#
#
# agent = RL.Agent(100, 0.99)
# in_a = torch.rand(5, device='cuda')
# in_b = torch.rand(5, device='cuda')
# loss_a = - torch.sum(agent.Actor(in_a))
# loss_b = - torch.sum(agent.Actor(in_b))
# loss_c = F.mse_loss(agent.Critic(in_a), agent.Critic(in_b))
# loss_d = F.mse_loss(agent.Critic(in_a), agent.Critic(in_b))
# stack_1 = loss_a + loss_b
# stack_2 = loss_c + loss_d
# print(stack_1.item(), stack_2.item())







# data = '1e-4\n'
# print(data[:-1])
# print(torch.zeros(1, device='cuda'))

# agent = RL.Agent()
# state = torch.tensor([3112., 61., 0., 0., 0.], device='cuda')
# action = 0
# next_state = torch.tensor([3122., 61., 0., 0., 0.], device='cuda')
# env = RL.Environment()
# print(env.State_diff_check(state, next_state, action))



# #a = torch.tensor([1242., 61., 0., 0., 0.], device='cuda')
# detected = [['Branch', [650, 65]], ['Branch', [650, 205]], ['Branch', [650, 345]], ['Branch', [650, 485]], ['Branch', [650, 625]]]
# next_state = env.Step(detected)
# print('next_state', next_state)




# EPSILON1 = 0.01
# EPSILON2 = 0.99
#
# state = torch.tensor([1242., 61., 0., 0., 0.], device='cuda')
# agent = RL.Agent()
# action = agent.Actor(state)
# print('action : ', action)
# random = np.random.uniform(0, 1)
# print('random : ', random)
# action = action.argmax() if EPSILON1 > random else action.argmin()
# print('action_chosen : ', action)
# action = action.argmax() if EPSILON2 > random else action.argmin()
# print('action_chosen : ', action)
#
# prob = torch.tensor([2345., 564.], device='cuda')
# print('prob : ', prob)
# print('soft : ', F.softmax(prob, dim=0))
# m = nn.LogSoftmax(dim=0)
# action = m(prob)
# print('log_soft : ', action)
# print('log_soft : ', action.argmin().item())
# print('log_soft : ', action.argmax().item())