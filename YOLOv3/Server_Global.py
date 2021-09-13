import os.path as op
import socket
import sys
import threading
import pickle
import torch

import Main_Header as Main
import RL_Header as RL

from torch.utils.tensorboard import SummaryWriter
from threading import Lock
from datetime import datetime


# 호스트, 포트번호, 동시접속제한
HOST = '210.110.39.196'
PORT = 9999
SIMULTANEOUS_LIMIT = 2
# 글로벌 에이전트, 글로벌 플로팅 모듈 인스턴스, 워커 인스턴스
SERVER_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
GLOBAL_AGENT = RL.Agent(0.001, 0.001, 0.001, 32, 1, 8)
TENSORBOARD = SummaryWriter(log_dir='runs/a3c')
WORKER = []
LINK = [0, 0]


def handle_worker(client_socket, address, mutex):
    """소켓 핸들러"""
    # 워커 주소수집
    if address[0] not in WORKER:
        WORKER.append(address[0])
        print(f'<<\t<< Newly connected with{WORKER} >>\t>>')

    # 워커 인덱스 추출
    worker_idx = WORKER.index(address[0])
    LINK[worker_idx] += 1

    # 워커 접속정보 출력
    print(f'[Worker <<{worker_idx}>> {address[0]} connected]')
    print(f'connection cycle <<{LINK[worker_idx]}>>')

    # 배치데이터(메타(바이트크기)데이터, 배치데이터) 수신
    worker_meta = pickle.loads(client_socket.recv(1024))
    worker_data = []
    while worker_meta > 0:
        packet = client_socket.recv(10000)
        worker_meta -= sys.getsizeof(packet)
        worker_data.append(packet)

    # 수신 배치데이터 디코딩
    GLOBAL_AGENT.Batch = pickle.loads(b''.join(worker_data))['batch']

    # 임계영역 진입
    print(f'{address[0]}{address[0]}{address[0]}{address[0]}{address[0]}{address[0]}{address[0]}{address[0]}')
    mutex.acquire()

    #####################################################################################################
    # 신경망 업데이트
    state, v_value, action, reward, advantage, q_value = GLOBAL_AGENT.Variable_ready_for_TD_N_Parallel()
    GLOBAL_AGENT.Update_by_TD(state, v_value, action, reward, advantage, q_value)
    GLOBAL_AGENT.Step_stack += 1

    # 에피소드 종료 시, 학습정보 출력 및 텐서보드 플로팅
    print(f'Actor_loss\t{GLOBAL_AGENT.Actor_loss_stack/GLOBAL_AGENT.Step_stack}')
    print(f'Critic_loss\t{GLOBAL_AGENT.Critic_loss_stack/GLOBAL_AGENT.Step_stack}')
    print(f'Reward\t{GLOBAL_AGENT.Reward_stack/GLOBAL_AGENT.Step_stack}')
    TENSORBOARD.add_scalar('Actor_loss', GLOBAL_AGENT.Actor_loss_stack/GLOBAL_AGENT.Step_stack, sum(LINK))
    TENSORBOARD.add_scalar('Critic_loss', GLOBAL_AGENT.Critic_loss_stack/GLOBAL_AGENT.Step_stack, sum(LINK))
    TENSORBOARD.add_scalar('Reward', GLOBAL_AGENT.Reward_stack/GLOBAL_AGENT.Step_stack, sum(LINK))
    #####################################################################################################

    # 임계영역 탈출
    mutex.release()
    print(f'{address[0]}{address[0]}{address[0]}{address[0]}{address[0]}{address[0]}{address[0]}{address[0]}')
    print('update fin\n')

    # 송신 가중치데이터 인코딩
    message = pickle.dumps({
        'actor_weights': GLOBAL_AGENT.Actor.state_dict(),
        'critic_weights': GLOBAL_AGENT.Critic.state_dict(),
        'flag': 'ok'
    })
    message_meta = pickle.dumps(sys.getsizeof(message))

    # 가중치데이터(메타(바이트크기)데이터, 가중치데이터) 송신
    client_socket.sendall(message_meta)
    client_socket.sendall(message)

    # 글로벌 모델 저장
    torch.save(GLOBAL_AGENT.Actor.state_dict(), Main.MODEL_PATH+Main.TBA_A3C)
    torch.save(GLOBAL_AGENT.Critic.state_dict(), Main.MODEL_PATH+Main.TBC_A3C)


def accept():
    """소켓 리스너"""
    SERVER_SOCKET.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    SERVER_SOCKET.bind((HOST, PORT))
    print('waiting')
    SERVER_SOCKET.listen(2)

    while True:
        try:
            client_socket, address = SERVER_SOCKET.accept()
        except KeyboardInterrupt:
            SERVER_SOCKET.close()
            print("Keyboard interrupt")
        thread = threading.Thread(target=handle_worker, args=(client_socket, address, mutex))
        thread.daemon = True
        thread.start()


if __name__ == '__main__':
    # 글로벌컴퓨터 로컬가중치 불러오기
    if op.isfile(Main.MODEL_PATH+Main.TBA_A3C) and op.isfile(Main.MODEL_PATH+Main.TBC_A3C):
        GLOBAL_AGENT.Actor.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBA_A3C))
        GLOBAL_AGENT.Critic.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBC_A3C))

    # 병행성 제어
    mutex = Lock()
    # 소켓 리스닝
    accept()
