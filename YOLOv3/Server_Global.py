import socket
import argparse
import threading
import time
import pickle
import torch

import Main_Header as Main
import RL_Header as RL

from torch.utils.tensorboard import SummaryWriter

# 호스트, 포트번호
HOST = '210.110.39.196'
PORT = 9999
# 글로벌 에이전트, 글로벌 플로팅 모듈 인스턴스
GLOBAL_AGENT = RL.Agent(0.001, 0.001, 0.001, 32, 1, 8)
tensorboard = SummaryWriter(log_dir='runs/a3c')


def handle_worker(client_socket, address):
    """소켓 핸들러"""
    print(f'접속 클라이언트 주소 {address}')
    worker_data = pickle.loads(client_socket.recv(10000))
    GLOBAL_AGENT.Batch = worker_data['batch']
    state, v_value, action, reward, advantage, q_value = GLOBAL_AGENT.Variable_ready_for_TD_N_Parallel()
    GLOBAL_AGENT.Update_by_TD(state, v_value, action, reward, advantage, q_value)
    GLOBAL_AGENT.Step_stack += 1
    # 에피소드 종료 시, 플로팅
    tensorboard.add_scalar('Actor_loss', GLOBAL_AGENT.Actor_loss_stack/GLOBAL_AGENT.Step_stack)
    tensorboard.add_scalar('Critic_loss', GLOBAL_AGENT.Critic_loss_stack/GLOBAL_AGENT.Step_stack)
    tensorboard.add_scalar('Reward', GLOBAL_AGENT.Reward_stack/GLOBAL_AGENT.Step_stack)
    tensorboard.close()
    # 데이터(글로벌 신경망 가중치 및 플래그 변수) 전송 및 통신 종료
    message = {'actor_weights': GLOBAL_AGENT.Actor.state_dict(), 'critic_weights': GLOBAL_AGENT.Critic.state_dict(), 'flag': 'ok'}
    client_socket.sendall(pickle.dumps(message))
    client_socket.close()
    # 글로벌 모델 저장
    torch.save(GLOBAL_AGENT.Actor.state_dict(), Main.MODEL_PATH+Main.TBA_A3C)
    torch.save(GLOBAL_AGENT.Critic.state_dict(), Main.MODEL_PATH+Main.TBC_A3C)


def accept_func():
    """소켓 리스너"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    print('waiting')
    server_socket.listen(2)

    while True:
        try:
            client_socket, address = server_socket.accept()
        except KeyboardInterrupt:
            server_socket.close()
            print("Keyboard interrupt")
        # accept()함수로 입력만 받아주고 이후 알고리즘은 핸들러에게 맡긴다.
        thread = threading.Thread(target=handle_worker, args=(client_socket, address))
        thread.daemon = True
        thread.start()


if __name__ == '__main__':
    global server_socket
    accept_func()
