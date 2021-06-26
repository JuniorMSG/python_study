"""
Author  : SG
DATE    : 2021-05-27

03. 제어문 , 반복문

03_01. if       (제어문)
03_02. while    (반복문)
★ 03_03. for   (반복문)
03_04. Question algorithm

★ 03_03. for   (반복문)
    01. 기본 사용방법
        01_01. 기본구조
        01_02. 리스트
        01_03. 문자열
        01_04. range()

    02. 진행 및 종료에 관한 구문
        02_01. 진행 : continue
        02_02. 종료 : break

    03. 예제
        03_01. 백준 2739번 문제
        03_02. 백준 10950번 문제

"""

"""
        01_01. 기본구조
            기본구조 
            for 변수 in 리스트(튜플, 문자열등):
                수행할 문장~   
        01_02. 리스트
        01_03. 문자열
        01_04. range()
"""
print("\n", "=" * 5, "01. 기본 사용방법", "=" * 5)
print("\n", "=" * 3, "01_02. 리스트", "=" * 3)
# 리스트 사용
lst_01 = [1, 2, 3]
for data in lst_01:
    print(data)

# 2중 리스트 사용
lst_02 = [[1,2], [3,4], [5,6]]
for data1, data2 in lst_02:
    print("data1 :", data1)
    print("data2 :", data2)

print("\n", "=" * 3, "01_03. 문자열", "=" * 3)
str_01 = "test"
# 문자열 사용
for data in str_01:
    print(data)

print("\n", "=" * 3, "01_04. range", "=" * 3)
# range 함수사용
for data in range(10):
    print(data)

"""
    02. 진행 및 종료에 관한 구문
        02_01. 진행 : continue
        02_02. 종료 : break
"""

print("\n", "=" * 5, "02. 진행 및 종료에 관한 구문", "=" * 5)

# 3미출력 5에서 종료
for data in range(10):
    # 02_01 아무것도 하지 않고 다음으로 진행
    if data == 3:
        print("\n", "=" * 3, "02_01. 진행 : continue", "=" * 3)
        continue

    # 02_02. 종료 : break
    if data == 5:
        print("\n", "=" * 3, "02_02. 종료 : break", "=" * 3)
        break
    print(data)


"""
    03. 예제
        03_01. 백준 2739번 문제
        03_02. 백준 10950번 문제
"""

print("\n", "=" * 5, "03. 예제", "=" * 5)

"""
03_01. 
백준 2739번 문제
N을 입력받은 뒤, 구구단 N단을 출력하는 프로그램을 작성하시오. 출력 형식에 맞춰서 출력하면 된다.

입력
첫째 줄에 N이 주어진다. N은 1보다 크거나 같고, 9보다 작거나 같다.

출력
출력형식과 같게 N*1부터 N*9까지 출력한다.
"""
def Q_2739():
    dan = int(input("단을 입력하세요 : "))
    for i in range(1, 10):
        print(dan, "*", i, "=", dan*i)


print("\n", "=" * 3, "03_01. 백준 2739번 문제", "=" * 3)
Q_2739()

"""
def Q_10950():
백준 10950번 문제
두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

입력
첫째 줄에 테스트 케이스의 개수 T가 주어진다.
각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)

출력
각 테스트 케이스마다 A+B를 출력한다.
"""
def Q_10950():
    testcase = int(input("입력받을 케이스 개수 : "))
    listCase = []

    for i in range(0, testcase):
        listCase.append(input("두 정수를 입력하세요 : "))

    for i, item in enumerate(listCase):
        num = item.split()
        print(int(num[0]) + int(num[1]))
    return


print("\n", "=" * 3, "03_01. 백준 10950번 문제", "=" * 3)
Q_10950()