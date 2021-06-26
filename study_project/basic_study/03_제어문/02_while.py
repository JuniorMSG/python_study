"""
Author  : SG
DATE    : 2021-05-27

03_ 제어문 , 반복문

03_01. if (제어문)
★ 03_02. while (반복문)
03_03. for
03_04. Question algorithm

★ 03_02. while (반복문)
    01. 반복문이란?
    02. while문
        02_01. 기본 사용법
        02_02. break
    03. 무한루프
        03_01. 만드는법
        03_02. 빠져 나가는법 
"""

"""
    01. 반복문이란?
    제어문의 한 종류로써
    프로그램 내에서 똑같은 명령을 일정 횟수만큼 반복하여 수행하도록 제어하는 명령문 
"""

"""
    03. 무한루프
    무한하게 루프를 반복한다는 의미로 많은 프로그램에서 사용합니다.

    03_01. 무한루프를 사용하는 이유
        자판기를 예로들면
        자판기를 사용하려고 2천원을 넣었습니다. 음료수 천원짜리를 뽑아서 천원이 남았습니다.
        추가로 돈을 넣을지, 음료수를 구매할지, 잔돈을 받을지 n번을 작업해야 되는데 n번이 몇번인지 모릅니다
        이런식으로 무한루프를 사용하여 특정 조건이 만족할때까지 프로그램을 대기 시키는 역할을 수행하게 됩니다.

    03_02. 사용하는 방법
        while True:
        num_while = num_while + 1
        if num_while == 1000:
            break

    03_03. 빠져나가는 방법
        Ctrl + C

    03_04. 활용방법 테스트용 백준 문제풀이
        코드첨부
"""

print("\n", "=" * 5, "03. 무한루프", "=" * 5)
print("\n", "=" * 3, "03_01. 사용하는 방법", "=" * 3)

num_while = 0
while True:
    num_while = num_while + 1
    if num_while == 1000:
        break
print(num_while)

print("\n", "=" * 3, "03_02. 빠져 나가는법", "=" * 3)

num_while = 0
# while True:
#     num_while = num_while + 1
#     print(num_while)
# Ctrl + C 키를 누르면 무한루프에서 빠져나간다.


print("\n", "=" * 3, "03_03. 활용 방법_", "=" * 3)
"""
https://www.acmicpc.net/problem/10952
백즌 10952번 문제 파이썬
두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

입력
입력은 여러 개의 테스트 케이스로 이루어져 있다.
각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)
입력의 마지막에는 0 두 개가 들어온다.

출력
각 테스트 케이스마다 A+B를 출력한다.
"""


def Q_10952():
    while True:
        num1, num2 = map(int, input().split())
        if num1 == 0 and num2 == 0:
            break
        else:
            print(num1 + num2)
    return


# Q_10952()


"""
https://www.acmicpc.net/problem/1110
백준 1110번 문제 파이썬
0보다 크거나 같고, 99보다 작거나 같은 정수가 주어질 때 다음과 같은 연산을 할 수 있다. 
먼저 주어진 수가 10보다 작다면 앞에 0을 붙여 두 자리 수로 만들고, 각 자리의 숫자를 더한다. 
그 다음, 주어진 수의 가장 오른쪽 자리 수와 앞에서 구한 합의 가장 오른쪽 자리 수를 이어 붙이면 새로운 수를 만들 수 있다. 
다음 예를 보자. 26부터 시작한다. 2+6 = 8이다. 새로운 수는 68이다. 6+8 = 14이다. 새로운 수는 84이다. 8+4 = 12이다. 새로운 수는 42이다. 4+2 = 6이다. 새로운 수는 26이다.

위의 예는 4번만에 원래 수로 돌아올 수 있다. 따라서 26의 사이클의 길이는 4이다.
N이 주어졌을 때, N의 사이클의 길이를 구하는 프로그램을 작성하시오.

입력
첫째 줄에 N이 주어진다. N은 0보다 크거나 같고, 99보다 작거나 같은 정수이다.

출력
첫째 줄에 N의 사이클 길이를 출력한다.

예제 입력 1 
26
예제 출력 1 
4
"""


def Q_1110():
    num1 = int(input())
    sp1 = int(num1 / 10)
    sp2 = num1 % 10
    sp3 = sp1 + sp2

    n = 0
    while True:
        sp3 = sp1 + sp2
        if sp3 >= 10:
            sp3 = sp3 % 10
        sp1, sp2 = sp2, sp3
        n = n + 1
        if sp1 * 10 + sp2 == num1:
            break

    print(n)
    return


Q_1110()

