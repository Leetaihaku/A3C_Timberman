import socket
import sys
import time

import torch
import pickle
from datetime import datetime

HOST = '210.110.39.196'
PORT = 9999


def transmit_batch(batch):
    # 클라이언트 소켓 생성 및 서버연결
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    # 송신 배치데이터 인코딩 및 송신
    client_socket.sendall(pickle.dumps(batch))

    # 가중치데이터(메타(바이트크기)데이터, 가중치데이터) 수신 및 디코딩
    global_data = pickle.loads(client_socket.recv(1000000))

    # 소켓통신 종료 및 수신 가중치데이터 반환
    client_socket.close()
    return global_data['actor_weights'], global_data['critic_weights']
