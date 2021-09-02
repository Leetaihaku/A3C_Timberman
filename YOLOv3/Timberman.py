import sys
import Main_Header as Main

if __name__ == '__main__':
    # 모드 선택
    select = int(input('1. train / 2. test / 3. transfer : '))
    if select == 1 or select == 3:
        Main.trainer(True if select == 1 else False)
        print('일반/전이훈련 종료')
    elif select == 2:
        Main.tester()
        print('테스트 종료')
    else:
        print('선택 에러')