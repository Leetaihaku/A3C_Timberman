import socket
import torch
import pickle

HOST = '210.110.39.196'
PORT = 9999


# 데이터 송수신 양식
class data:
    def __init__(self, state, action, reward, next_state, final_flag):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.final_flag = final_flag


def transmit_batch(state, action, reward, next_state, final_flag):
    # 클라이언트 소켓 생성 및 서버연결
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    # 데이터 전송
    message = data(state, action, reward, next_state, final_flag)
    client_socket.sendall(pickle.dumps(message))
    receive_data = pickle.loads(client_socket.recv(4096))

    # 모델 업데이트 완료 신호 + 접속 종료 및 통신 단절
    while True:
        if receive_data == 'ok':
            client_socket.close()
            break
    return
