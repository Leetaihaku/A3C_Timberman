import subprocess
import RL_Header as RL
import pyautogui
import time
import keyboard

from torch.utils.tensorboard import SummaryWriter

# BlueStack(앱 플레이어) 실행경로
ADDRESS_BlueStack = "C:\\Program Files\\BlueStacks\\Bluestacks.exe"
# 이미지, 디바이스, 감지모델 경로
WEBCAM_PATH = DEVICE_PATH = '0'
MODEL_PATH = "C:\\RL_Timberman\\YOLOv3\\"
BEST = 'best.pt'

# 학습모델, 배치파일 경로(훈련용)
TBA = 'TBA.pt'
TBC = 'TBC.pt'
DETECT_BAT = 'Detect.bat'
DETECT_BAT_EXE = 'Detect_exe.bat'
INFO = 'Learning_info.txt'

# 학습모델, 배치파일 경로(테스터용)
TBA_TEST = 'TBA_TEST.pt'
TBC_TEST = 'TBC_TEST.pt'
DETECT_BAT_TEST = 'Detect_test.bat'
DETECT_BAT_EXE_TEST = 'Detect_exe_test.bat'
INFO_TEST = 'Test_info.txt'

# 학습모델, 배치파일 경로(전이용)
TBA_TRANSFER = 'TBA_TRANSFER.pt'
TBC_TRANSFER = 'TBC_TRANSFER.pt'
DETECT_BAT_TRANSFER = 'Detect_transfer.bat'
DETECT_BAT_EXE_TRANSFER = 'Detect_exe_transfer.bat'
INFO_TRANSFER = 'Transfer_info.txt'


def new_or_load(mode):
    """학습 새로하기 또는 이어하기 → 하이퍼파라미터 입력받기"""
    while True:
        select = input('학습 실행(1. 새로하기, 2. 이어하기) : ')
        if select == '1':
            # 학습횟수 입력
            epoch = input('EPOCH : ')
            epoch = 1000 if epoch == '' else int(epoch)
            # 불확실성
            epsilon = input('EPSILON : ')
            epsilon = 0.999 if epsilon == '' else float(epsilon)
            # 불확실성 감소팩터
            epsilon_discount = input('EPSILON_DISCOUNT : ')
            epsilon_discount = 1e-3 if epsilon_discount == '' else float(epsilon_discount)
            # 학습률
            learning_rate = input('LEARNING_RATE : ')
            learning_rate = 1e-3 if learning_rate == '' else float(learning_rate)
            # 노드 수
            node = input('NODE : ')
            node = 32 if node == '' else int(node)
            # 노드 수
            step_mode = input('STEP_MODE : ')
            step_mode = 1 if step_mode == '' else int(step_mode)
            # 노드 수
            batch_size = input('BATCH_SIZE : ')
            batch_size = 8 if batch_size == '' else int(batch_size)
            return epoch, epoch, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size
        elif select == '2':
            # 학습정보 메모장 읽기
            with open(MODEL_PATH + (INFO if mode else INFO_TRANSFER), 'r') as learning_info:
                epoch_origin = int(learning_info.readline())
                epoch = int(learning_info.readline())
                epsilon = float(learning_info.readline())
                epsilon_discount = float(learning_info.readline())
                learning_rate = float(learning_info.readline())
                node = int(learning_info.readline())
                step_mode = int(learning_info.readline())
                batch_size = int(learning_info.readline())
            return epoch_origin, epoch, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size
        else:
            print('잘못된 선택입니다.')


def new_or_load_test():
    """테스트용 :: 학습 새로하기 또는 이어하기 → 하이퍼파라미터 입력받기"""
    while True:
        select = input('(테스트 실행) 1.새로하기 / 2.이어하기 : ')
        if select == '1':
            # 학습횟수 입력
            epoch = input('EPOCH : ')
            epoch = 1000 if epoch == '' else int(epoch)
            return epoch, epoch
        elif select == '2':
            # 학습정보 메모장 읽기
            with open(MODEL_PATH + INFO_TEST, 'r') as test_info:
                epoch_origin = int(test_info.readline())
                epoch = int(test_info.readline())
            return epoch_origin, epoch
        else:
            print('잘못된 선택입니다.')


def learning_sequence(mode, cmd_init, tensorboard, episode):
    """학습 진행"""
    # 학습 하이퍼파라미터 읽어오기
    epoch_origin, epoch, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size = lr_params_read(mode)
    # 디렉토리 변경 & cmd → 아나콘다 활성화 & 실행 & 학습결과 반환데이터 수령
    # (Windows bat. file execute)
    # 불확실성 값 → 실행 배치파일 내 전달
    ready_exeute_bat(mode, cmd_init, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size)
    # 서브프로세스 실행
    data = subprocess.run(MODEL_PATH + (DETECT_BAT_EXE if mode else DETECT_BAT_EXE_TRANSFER), shell=True, capture_output=True, text=True)
    # 반환데이터 리스트화
    print(data)
    data = data.stdout.split('\n')
    # 학습데이터 리스트화
    data = list(map(float, data[-2].split(' ')))
    # 학습결과 및 종료메세지 출력
    print_learning_result(data, epsilon, learning_rate, episode)
    # 학습결과 텐서보드 플로팅
    tensorboard_plotting(tensorboard, data, episode)
    # 학습정보 갱신 및 저장
    update_lr_params_write(mode, epoch_origin, epoch, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size)
    return


def test_sequence(cmd_test_init):
    """테스트 진행"""
    epoch_origin, epoch = test_params_read()
    ready_test_execute_bat(cmd_test_init)
    data = subprocess.run(MODEL_PATH + DETECT_BAT_EXE_TEST, shell=True, capture_output=True, text=True)
    print(data)
    update_test_params_write(epoch_origin, epoch)
    return


def init_lr_params_write(mode, epoch_origin, epoch, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size):
    """학습 하이퍼파라미터 최초 저장"""
    with open(MODEL_PATH + (INFO if mode else INFO_TRANSFER), 'w') as learning_info:
        learning_info.write(f'{str(epoch_origin)}\n{str(epoch)}\n{str(epsilon)}\n{str(epsilon_discount)}\n{str(learning_rate)}\n{str(node)}\n{str(step_mode)}\n{str(batch_size)}\n')
    return


def init_test_params_write(epoch_origin, epoch):
    """테스트용 :: 학습 하이퍼파라미터 최초 저장"""
    with open(MODEL_PATH + INFO_TEST, 'w') as test_info:
        test_info.write(f'{str(epoch_origin)}\n{str(epoch)}\n')
    return


def lr_params_read(mode):
    """학습 하이퍼파라미터 읽기"""
    with open(MODEL_PATH + (INFO if mode else INFO_TRANSFER), 'r') as learning_info:
        epoch_origin = int(learning_info.readline())
        epoch = int(learning_info.readline())
        epsilon = float(learning_info.readline())
        epsilon_discount = float(learning_info.readline())
        learning_rate = float(learning_info.readline())
        node = int(learning_info.readline())
        step_mode = int(learning_info.readline())
        batch_size = int(learning_info.readline())
    return epoch_origin, epoch, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size


def test_params_read():
    """테스트용 :: 학습 하이퍼파라미터 읽기"""
    with open(MODEL_PATH + INFO_TEST, 'r') as test_info:
        epoch_origin = int(test_info.readline())
        epoch = int(test_info.readline())
    return epoch_origin, epoch


def update_lr_params_write(mode, epoch_origin, epoch, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size):
    """학습 하이퍼파라미터 개정 및 저장"""
    epoch -= 1
    epsilon -= epsilon_discount if epsilon >= RL.EPSILON_LOWER_LIMIT else 0
    with open(MODEL_PATH + (INFO if mode else INFO_TRANSFER), 'w') as learning_info:
        learning_info.write(f'{str(epoch_origin)}\n{str(epoch)}\n{str(epsilon)}\n{str(epsilon_discount)}\n{str(learning_rate)}\n{str(node)}\n{str(step_mode)}\n{str(batch_size)}\n')
    return


def update_test_params_write(epoch_origin, epoch):
    """테스트용 :: 학습 하이퍼파라미터 개정 및 저장"""
    epoch -= 1
    with open(MODEL_PATH + INFO_TEST, 'w') as test_info:
        test_info.write(f'{str(epoch_origin)}\n{str(epoch)}\n')
    return


def cmd_init(mode):
    """실행 배치파일 전달명령어 원본 추출"""
    with open(MODEL_PATH + (DETECT_BAT if mode else DETECT_BAT_TRANSFER), 'r') as Detect_bat:
        total = Detect_bat.readlines()
    return total[0] + total[1]


def cmd_test_init():
    """테스트용 :: 실행 배치파일 전달명령어 원본 추출"""
    with open(MODEL_PATH + DETECT_BAT_TEST, 'r') as Detect_bat_test:
        total = Detect_bat_test.readlines()
    return total[0] + total[1]


def cmd_final(cmd_init, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size):
    """최종 실행배치파일 내 실행명령어 저장"""
    return cmd_init + f' --epsilon {epsilon} --epsilon_discount {epsilon_discount} --learning_rate {learning_rate} --node {node} --step_mode {step_mode} --batch_size {batch_size}\n'


def cmd_test_final(cmd_test_init):
    """테스트용 :: 최종실행배치파일 내 실행명령어 저장"""
    return cmd_test_init


def ready_exeute_bat(mode, cmd_init, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size):
    """불확실성 값 → 실행 배치파일 내 전달"""
    with open(MODEL_PATH + (DETECT_BAT_EXE if mode else DETECT_BAT_EXE_TRANSFER), 'w') as Detect_bat_exe:
        final = cmd_final(cmd_init, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size)
        Detect_bat_exe.write(final)
    return


def ready_test_execute_bat(cmd_test_init):
    """테스트용 :: 불확실성 값 → 실행 배치파일 내 전달"""
    with open(MODEL_PATH + DETECT_BAT_EXE_TEST, 'w') as Detect_bat_exe_test:
        test_final = cmd_test_final(cmd_test_init)
        Detect_bat_exe_test.write(test_final)
    return


def print_learning_result(data, epsilon, learning_rate, i):
    """학습정보 출력"""
    if i == 1:
        print(f'{i}st Training result')
    elif i == 2:
        print(f'{i}nd Training result')
    elif i == 3:
        print(f'{i}rd Training result')
    else:
        print(f'{i}th Training result')
    print(f'Actor_loss\t{data[0]}')
    print(f'Critic_loss\t{data[1]}')
    print(f'Reward\t{data[2]}')
    print(f'epsilon\t{epsilon}')
    print(f'learning_rate\t{learning_rate}')


def tensorboard_plotting(tensorboard, data, i):
    """텐서보드 플로팅"""
    tensorboard.add_scalar('Actor_loss', float(data[0]), int(i))
    tensorboard.add_scalar('Critic_loss', float(data[1]), int(i))
    tensorboard.add_scalar('Reward', float(data[2]), int(i))


def reboot_game():
    """에피소드 10회당 게임 리부트"""
    keyboard.press_and_release('esc')
    time.sleep(1)
    pyautogui.moveTo(x=1665, y=995)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(x=1040, y=75)
    pyautogui.click()
    time.sleep(1)
    keyboard.press_and_release('t')
    time.sleep(1)
    keyboard.press_and_release('F11')


def reboot_program():
    """에피소드 50회당 프로그램 리부트"""
    keyboard.press_and_release('esc')
    time.sleep(1)
    pyautogui.moveTo(x=1630, y=15)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(x=950, y=540)
    pyautogui.click()
    time.sleep(15)
    pyautogui.moveTo(x=665, y=1030)
    pyautogui.click()
    time.sleep(45)
    pyautogui.moveTo(x=1490, y=120)
    pyautogui.click()
    time.sleep(1)
    keyboard.press_and_release('t')
    time.sleep(1)
    keyboard.press_and_release('F11')


def trainer(mode):
    """일반훈련 / 전이훈련"""
    # 학습 하이퍼파라미터 가져오기
    epoch_origin, epoch, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size = new_or_load(mode)
    # 학습 하이퍼파라미터 저장
    init_lr_params_write(mode, epoch_origin, epoch, epsilon, epsilon_discount, learning_rate, node, step_mode, batch_size)
    # 학습 그래프 모듈 실행 & 불확실성 감소 값
    tensorboard = SummaryWriter(log_dir=('runs/trainer' if mode else 'runs/transfer'))
    # 실행 배치파일 전달명령어 원본 추출
    init = cmd_init(mode)
    # 학습 진행
    for episode in range(epoch_origin-epoch+1, epoch_origin+1):
        learning_sequence(mode, init, tensorboard, episode)
        if episode % 10 == 0:
                reboot_program() if episode % 50 == 0 else reboot_game()
    # 플로팅 인스턴스 제거
    tensorboard.close()
    return


def tester():
    """테스트"""
    ############################################################################################
    # 1-상태 6-행동생성 및 시행(새 detect.py 제작도 고려)
    # 1. 현재상태 -> 상태발생기 활용 -> 6단계 미래 생성
    # 2. 6단계의 미래에 대한 순차적 행동시퀀스 생성 및 실행
    # 3. t+1 == t+2인 상태이미지를 현재상태로 재정의
    # 4. 앞선 작업 반복
    ############################################################################################
    # 테스트 하이퍼파라미터 가져오기
    epoch_origin, epoch = new_or_load_test()
    # 테스트 하이퍼파라미터 저장
    init_test_params_write(epoch_origin, epoch)
    # 실행 배치파일 전달명령어 원본 추출
    init = cmd_test_init()
    # 학습 진행
    for episode in range(epoch_origin - epoch + 1, epoch_origin + 1):
        test_sequence(init)
        if episode % 10 == 0:
            reboot_program() if episode % 50 == 0 else reboot_game()
    return
