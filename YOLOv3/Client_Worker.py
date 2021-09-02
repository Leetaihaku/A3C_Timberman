import socket
import argparse

HOST = '210.110.39.196'
PORT = 9999

# 접속하고 싶은 ip와 port를 입력받는 클라이언트 코드를 작성해보자.
# 접속하고 싶은 포트를 입력한다.
if __name__ == '__main__':
    # parser와 관련된 메서드 정리된 블로그 : https://docs.python.org/ko/3/library/argparse.html
    # description - 인자 도움말 전에 표시할 텍스트 (기본값: none)
    # help - 인자가 하는 일에 대한 간단한 설명.
    # nargs - 소비되어야 하는 명령행 인자의 수. -> '+'로 설정 시 모든 명령행 인자를 리스트로 모음 + 없으면 경고
    # required - 명령행 옵션을 생략 할 수 있는지 아닌지 (선택적일 때만).
    parser = argparse.ArgumentParser(description="\nJoo's client\n-p port\n-i host\n-s string")
    parser.add_argument('-p', help="port")
    parser.add_argument('-i', help="host", required=True)
    parser.add_argument('-u', help="user", required=True)
    args = parser.parse_args()
    host = args.i
    user = str(args.u)
    try:
        port = int(args.p)
    except:
        pass #IPv4 체계, TCP 타입 소켓 객체를 생성
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 지정한 host와 prot를 통해 서버에 접속합니다.
    client_socket.connect((HOST, PORT))
    client_socket.sendall(user.encode())
    receive_data = client_socket.recv(1024)
    print("받은 데이터는 \"", receive_data.decode(), "\" 입니다.", sep="")
    # 소켓을 닫는다.
    client_socket.close()
    print("접속을 종료합니다.")