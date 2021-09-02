import argparse
import torch
import RL_Header as RL
import random
import Main_Header as Main

def init_state_generator():
    """랜덤 초기상태(총가지 2개)"""
    # 5, 4단에서 선택 1
    front = random.choice(['41', '42', '51', '52'])
    # 3, 2단에서 선택 2(선택 1에 대한 조건부)
    if front == '41':
        rear = random.choice(['31', '21', '22', '11', '12'])
    elif front == '42':
        rear = random.choice(['32', '21', '22', '11', '12'])
    elif front == '51':
        rear = random.choice(['41', '31', '32', '21', '22', '11', '12'])
    else:
        rear = random.choice(['42', '31', '32', '21', '22', '11', '12'])
    # 나뭇가지 상태생성
    branch = float(front+rear)
    return torch.tensor([branch, random.choice([61., 62.]), 0., 0., 0.], device='cuda'), False

def state_generator(state, action):
    """다음상태 발생"""
    state_tolist = state.tolist()
    player = 62. if action else 61.
    branch = str(int(state_tolist[0]))
    # 사망 프로세스
    if branch[:2] in ['51', '61'] and action == 0 or branch[:2] == ['52', '62'] and action == 1:
        return torch.tensor([0., 0., 0., 0., 1.], device='cuda'), True
    # 생존 프로세스
    else:
        if branch[0] == '6':
            if len(branch) > 2:
                branch = branch[2:]
            else:
                return torch.tensor([random.choice([11., 12.]), player, 0., 0., 0.], device='cuda'), False

        add = ''
        for i in range(len(branch) >> 1):
            add += '10'
        branch = str(int(branch) + int(add))
        if branch[len(branch)-2:] not in ['11', '12'] and len(branch) <= 4 and random.uniform(0, 1) > 0.333:
            branch = branch+'11' if branch[len(branch)-2:] == '21' else branch+'12' if branch[len(branch)-2:] == '22' else branch+random.choice(['11', '12'])
        return torch.tensor([float(branch), float(player), 0., 0., 0.], device='cuda'), False

if __name__ == '__main__':
    # 매개변수 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default='0.999', help='epsilon')
    parser.add_argument('--epsilon_discount', type=float, default='0.001', help='epsilon_discount')
    parser.add_argument('--learning_rate', type=float, default='0.001', help='learning_rate')
    parser.add_argument('--node', type=int, default='32', help='num of node')
    parser.add_argument('--step_mode', type=int, default=1, help='True : n-step, False : 0-step')
    parser.add_argument('--batch_size', type=int, default=8, help='N-step batch_size')
    obj = parser.parse_args()

    # 에이전트 생성
    env = RL.Environment()
    agent = RL.Agent(obj.epsilon, obj.epsilon_discount, obj.learning_rate, obj.node, bool(obj.step_mode), obj.batch_size)

    # 사전 신경망 설정
    agent.Start('test')

    # 훈련
    state, done = init_state_generator()
    while True:
        action = agent.Action(state, 'test')
        next_state, done = state_generator(state, action)
        reward = env.Reward(state, done)
        # TD-(N)
        if agent.Step_mode:
            agent.Batch.append([state.detach(), reward, next_state.detach(), done])
            if done:
                for i in range(len(agent.Batch)):
                    State, V_value, Reward, Advantage, Q_value = agent.Variable_ready_for_TD_N()
                    agent.Update_by_TD(State, V_value, Reward, Advantage, Q_value)
                    # 완료된 배치 대기열 큐 내 상태 제거
                    agent.Batch.popleft()
            # 배치크기 충족 시, Advantage, Q_value 계산 및 업데이트
            elif len(agent.Batch) == agent.Batch_size:
                # Advantage, Q_value 계산 및 업데이트
                State, V_value, Reward, Advantage, Q_value = agent.Variable_ready_for_TD_N()
                agent.Update_by_TD(State, V_value, Reward, Advantage, Q_value)
                # 완료된 배치 대기열 큐 내 상태 제거
                agent.Batch.popleft()
        # TD-(0)
        else:
            v_v, adv, q_v = agent.Variable_ready_for_TD_0(state, reward, next_state, done)
            agent.Update_by_TD(state, v_v, reward, adv, q_v)
        agent.Step_stack += 1
        # RL.Print_all(state, action, reward, next_state, v_v, n_v, q_v, adv)
        if done:
            break
        else:
            state = next_state
    
    # 저장
    torch.save(agent.Actor.state_dict(), Main.MODEL_PATH+Main.TBA_TEST)
    torch.save(agent.Critic.state_dict(), Main.MODEL_PATH+Main.TBC_TEST)

    # 출력
    print(agent.Actor_loss_stack.item()/agent.Step_stack, agent.Critic_loss_stack.item()/agent.Step_stack, agent.Reward_stack.item())
