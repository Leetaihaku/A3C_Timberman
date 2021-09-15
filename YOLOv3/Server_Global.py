import os.path as op
import socket
import threading
import pickle
import torch

import Main_Header as Main
import RL_Header as RL

from torch.utils.tensorboard import SummaryWriter
from threading import Lock


# 호스트, 포트번호, 동시접속제한
HOST = '210.110.39.196'
PORT = 9999
# 글로벌 플로팅 지점 및 저장경로
GLOBAL_PLOT_POINT = 0
GLOBAL_PLOT_SAVE = 'global_plot.txt'
# 락, 병행접속 최대 가능 수
MUTEX = Lock()
SIMULTANEOUS_LIMIT = 2
# 글로벌 에이전트, 글로벌 플로팅 모듈 인스턴스, 워커 인스턴스
SERVER_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
GLOBAL_AGENT = RL.Agent(0.001, 0.001, 0.001, 32, 1, 8)
TENSORBOARD = SummaryWriter(log_dir='runs/a3c')
WORKER = []
LINK = [0, 0]


def handle_worker(client_socket, address):
    """소켓 핸들러"""
    # 워커 주소수집
    if address[0] not in WORKER:
        if len(WORKER) <= SIMULTANEOUS_LIMIT:
            WORKER.append(address[0])
            print(f'<<\t<< Newly connected with{WORKER} >>\t>>')
        else:
            print('동시접속 가능한 최대 Worker 수를 초과')
            return

    # 워커 인덱스 추출
    worker_idx = WORKER.index(address[0])
    LINK[worker_idx] += 1

    # 워커 접속정보 출력
    print(f'[Worker <<{worker_idx}>> {address[0]} connected]')
    print(f'connection cycle <<{LINK[worker_idx]}>>')

    # 배치데이터 수신 및 디코딩
    try:
        worker_data = pickle.loads(client_socket.recv(1000000))
    except:
        print('배치데이터 수신에러 발생')
        return

    # 임계영역 진입
    print('\nupdate start')
    MUTEX.acquire()
    if WORKER.index(address[0]):
        print('△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△△')
    else:
        print('▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲')
    #####################################################################################################
    # 신경망 업데이트
    GLOBAL_AGENT.Batch = worker_data
    state, v_value, action, reward, advantage, q_value = GLOBAL_AGENT.Variable_ready_for_TD_N_Parallel()
    GLOBAL_AGENT.Update_by_TD(state, v_value, action, reward, advantage, q_value)
    GLOBAL_AGENT.Step_stack += 1

    # 에피소드 종료 시, 학습정보 출력 및 텐서보드 플로팅 및 플로팅 지점 저장
    print(f'Actor_loss\t{GLOBAL_AGENT.Actor_loss_stack/GLOBAL_AGENT.Step_stack}')
    print(f'Critic_loss\t{GLOBAL_AGENT.Critic_loss_stack/GLOBAL_AGENT.Step_stack}')
    print(f'Reward\t{GLOBAL_AGENT.Reward_stack/GLOBAL_AGENT.Step_stack}')
    TENSORBOARD.add_scalar('Actor_loss', GLOBAL_AGENT.Actor_loss_stack/GLOBAL_AGENT.Step_stack, sum(LINK))
    TENSORBOARD.add_scalar('Critic_loss', GLOBAL_AGENT.Critic_loss_stack/GLOBAL_AGENT.Step_stack, sum(LINK))
    TENSORBOARD.add_scalar('Reward', GLOBAL_AGENT.Reward_stack/GLOBAL_AGENT.Step_stack, sum(LINK))
    with open(GLOBAL_PLOT_SAVE, 'w') as GP:
        for i in range(len(WORKER)):
            GP.write(f'{WORKER[i]}\n')
        GP.write(f'{LINK[0]}\n{LINK[1]}\n')
    #####################################################################################################

    # 임계영역 탈출
    if WORKER.index(address[0]):
        print('▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽▽')
    else:
        print('▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼')
    MUTEX.release()
    print('update finish\n')

    # 송신 가중치데이터 인코딩 및 송신
    client_socket.sendall(pickle.dumps({'actor_weights': GLOBAL_AGENT.Actor.state_dict(), 'critic_weights': GLOBAL_AGENT.Critic.state_dict()}))

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
        thread = threading.Thread(target=handle_worker, args=(client_socket, address))
        thread.daemon = True
        thread.start()


if __name__ == '__main__':
    # 이어하기에 따른 플로팅지점 불러오기
    while True:
        select = int(input('1. 새 학습 / 2. 이어 학습 : '))
        if select == 1:
            pass
            break
        elif select == 2:
            with open(GLOBAL_PLOT_SAVE, 'r') as GP:
                GLOBAL_PLOT_POINT = GP.readlines()
            WORKER.append(GLOBAL_PLOT_POINT[0])
            WORKER.append(GLOBAL_PLOT_POINT[1])
            LINK[0] = GLOBAL_PLOT_POINT[2]
            LINK[1] = GLOBAL_PLOT_POINT[3]
            break
        else:
            print('잘못된 선택입니다.')
            print('다시 선택하세요')

    # 글로벌컴퓨터 로컬가중치 불러오기
    if op.isfile(Main.MODEL_PATH+Main.TBA_A3C) and op.isfile(Main.MODEL_PATH+Main.TBC_A3C):
        GLOBAL_AGENT.Actor.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBA_A3C))
        GLOBAL_AGENT.Critic.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBC_A3C))

    # 소켓 리스닝
    accept()
