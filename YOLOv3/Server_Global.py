import socket
import argparse
import threading
import time
import pickle
import torch

import Main_Header as Main
import RL_Header as RL


HOST = '210.110.39.196'
PORT = 9999
GLOBAL_AGENT = RL.Agent(0.001, 0.001, 0.001, 32, 1, 8)


class data:
    """송수신 데이터 양식"""
    def __init__(self, state, action, reward, next_state, final_flag):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.final_flag = final_flag


def handle_worker(client_socket, address):
    """소켓 핸들러"""
    print(f'접속 클라이언트 주소 {address}')
    worker_data = pickle.loads(client_socket.recv(4096))
    GLOBAL_AGENT.Save_batch(worker_data.state, worker_data.action, worker_data.reward, worker_data.next_state)
    # 종료 시, 잔여 업데이트 진행
    if worker_data.final_flag:
        for batch_idx in range(len(GLOBAL_AGENT.Batch)):
            state, v_value, mini_action, reward, advantage, q_value = GLOBAL_AGENT.Variable_ready_for_TD_N_Parallel()
            GLOBAL_AGENT.Update_by_TD(state, v_value, mini_action, reward, advantage, q_value)
            GLOBAL_AGENT.Batch.popleft()
    # 진행 시, 배치사이즈 한도 내에서 업데이트 진행
    elif len(GLOBAL_AGENT.Batch) == GLOBAL_AGENT.Batch_size:
        state, v_value, mini_action, reward, advantage, q_value = GLOBAL_AGENT.Variable_ready_for_TD_N_Parallel()
        GLOBAL_AGENT.Update_by_TD(state, v_value, mini_action, reward, advantage, q_value)
        GLOBAL_AGENT.Batch.popleft()
    # 글로벌 신경망 저장
    torch.save(GLOBAL_AGENT.Actor.state_dict(), Main.MODEL_PATH+Main.TBA_A3C)
    torch.save(GLOBAL_AGENT.Actor.state_dict(), Main.MODEL_PATH+Main.TBC_A3C)
    client_socket.sendall(pickle.dumps('ok'))
    client_socket.close()


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
