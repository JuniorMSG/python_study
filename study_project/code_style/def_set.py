"""
    00. default
        00_01. 출력 유형 통일

"""

def print_set(data):
    print('='*5, data, '='*5)

def default_01():
    print('='*20, '00. default', '='*20)

    lst = []
    lst.append('00_01. 출력 유형 통일 ')

    for i in range(1, len(lst)+1):
        print(i, ' : ', lst[i-1])
    lst_cnt = int(input('숫자를 입력하세요 : '))

    print('='*5, lst[lst_cnt-1], '='*5)

    if lst_cnt == 1:
        print(1)
    elif lst_cnt == 2:
        print(1)
    elif lst_cnt == 3:
        print(1)
    elif lst_cnt == 4:
        print(1)
    else:
        print('선택항목은 없는 항목')



