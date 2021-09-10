import socket
import torch
import pickle

HOST = '210.110.39.196'
PORT = 9999


def transmit_batch(batch):
    # 클라이언트 소켓 생성 및 서버연결
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    # 데이터 전송
    message = {'batch': batch}
    client_socket.sendall(pickle.dumps(message))
    receive_data = pickle.loads(client_socket.recv(10000))

    # 모델 업데이트 완료 신호 + 접속 종료 및 통신 단절
    while True:
        if receive_data['flag'] == 'ok':
            client_socket.close()
            return receive_data['actor_weights'], receive_data['critic_weights']
