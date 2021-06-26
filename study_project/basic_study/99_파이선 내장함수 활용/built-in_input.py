""" NW
    subject
        built-in Function (내장 함수)
    topic
        input() : 사용자 입력 함수
    Describe
        input()은 사용자의 입력을 받는 함수이다.
        입력 받은 값을 전부 문자열로 취급하는 특징을 가지고 있다.

    Contents
        01. 사용법
            num = input()
        02. 타입 변환 사용법
            1. 타입 직접 지정     : num = int(input())
            2. 맵으로 타입 지정   : num_lst = list(map(int, input()))
"""

# 01. 사용법
def input_01():
    """
        01. 사용법
            num = input()
    """
    print("\n", "=" * 5, "01. 사용법", "=" * 5)
    num1 = input('숫자를 입력해주세요 - 1 : ')
    num2 = input('숫자를 입력해주세요 - 2 : ')

    print('무조건 문자열이다. type(num) : ', type(num1))

    # 문자열로 연산된다.
    sum_data = num1 + num2
    print('문자열로 연산된다 : ', sum_data)


# input_01()

# 02. 타입 변환 사용법
def input_02():
    """
    02. 타입 변환 사용법
        02_01. 타입 직접 지정     : num = int(input())
        02_02. 맵으로 타입 지정   : num_lst = list(map(int, input()))
    """

    print("\n", "=" * 5, "02. 타입 변환 사용법", "=" * 5)
    print("\n", "=" * 3, "02_01. 타입 직접 지정", "=" * 3)
    num1 = input('숫자를 입력해주세요 - 1 : ')
    num2 = input('숫자를 입력해주세요 - 2 : ')

    # 타입을 지정하는방법 - 1
    sum_data = int(num1) + int(num2)
    print('타입을 지정하는방법 - 1 ', sum_data)

    print("\n", "=" * 3, "02_02. 맵으로 타입 지정", "=" * 3)

    # 타입을 지정하는 map 함수
    num_lst = list(map(int, input('숫자를 공백으로 입력하세요 EX) 5 9 : ').split()))
    sum_data = 0

    # 내장함수 sum으로 합산
    print('맵으로 사용하는방법 ', sum(num_lst))

    # for문으로 합산
    for data in num_lst:
        sum_data += data
    print('맵으로 사용하는방법 : ', sum_data)
input_02()