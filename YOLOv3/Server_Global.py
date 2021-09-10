import socket
import threading
import pickle
import torch

import Main_Header as Main
import RL_Header as RL

from torch.utils.tensorboard import SummaryWriter

# 호스트, 포트번호
HOST = '210.110.39.196'
PORT = 9999
# 글로벌 에이전트, 글로벌 플로팅 모듈 인스턴스, 워커 인스턴스
SERVER_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
GLOBAL_AGENT = RL.Agent(0.001, 0.001, 0.001, 32, 1, 8)
TENSORBOARD = SummaryWriter(log_dir='runs/a3c')
WORKER = []
LINK = [0, 0]


def handle_worker(client_socket, address):
    """소켓 핸들러"""
    # 워커 주소수집
    if address not in WORKER:
        WORKER.append(address[0])

    # 워커 인덱스 추출
    worker_idx = WORKER.index(address[0])
    LINK[worker_idx] += 1

    # 워커 접속정보 출력
    print(f'\n[Worker <<{worker_idx}>> connected]')
    print(f'connection cycle <<{LINK[worker_idx]}>>')

    # 학습진행
    worker_data = pickle.loads(client_socket.recv(10000))
    GLOBAL_AGENT.Batch = worker_data['batch']
    state, v_value, action, reward, advantage, q_value = GLOBAL_AGENT.Variable_ready_for_TD_N_Parallel()
    GLOBAL_AGENT.Update_by_TD(state, v_value, action, reward, advantage, q_value)
    GLOBAL_AGENT.Step_stack += 1

    # 에피소드 종료 시, 학습정보 출력 및 텐서보드 플로팅
    print(f'Actor_loss\t{GLOBAL_AGENT.Actor_loss_stack/GLOBAL_AGENT.Step_stack}')
    print(f'Critic_loss\t{GLOBAL_AGENT.Critic_loss_stack/GLOBAL_AGENT.Step_stack}')
    print(f'Reward\t{GLOBAL_AGENT.Reward_stack/GLOBAL_AGENT.Step_stack}')
    TENSORBOARD.add_scalar('Actor_loss', GLOBAL_AGENT.Actor_loss_stack/GLOBAL_AGENT.Step_stack)
    TENSORBOARD.add_scalar('Critic_loss', GLOBAL_AGENT.Critic_loss_stack/GLOBAL_AGENT.Step_stack)
    TENSORBOARD.add_scalar('Reward', GLOBAL_AGENT.Reward_stack/GLOBAL_AGENT.Step_stack)

    # 데이터(글로벌 신경망 가중치 및 플래그 변수) 전송 및 통신 종료
    message = {'actor_weights': GLOBAL_AGENT.Actor.state_dict(), 'critic_weights': GLOBAL_AGENT.Critic.state_dict(), 'flag': 'ok'}
    client_socket.sendall(pickle.dumps(message))
    client_socket.close()

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
    # 글로벌컴퓨터 로컬가중치 불러오기
    GLOBAL_AGENT.Actor.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBA_A3C))
    GLOBAL_AGENT.Critic.load_state_dict(torch.load(Main.MODEL_PATH+Main.TBC_A3C))

    # 소켓 리스닝
    accept()
