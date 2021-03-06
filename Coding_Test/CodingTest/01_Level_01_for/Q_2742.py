"""
    https://www.acmicpc.net/step/3
    Subject : Coding Test For
"""

"""
    https://www.acmicpc.net/problem/2742
    
    def Q_2742():
    자연수 N이 주어졌을 때, N부터 1까지 한 줄에 하나씩 출력하는 프로그램을 작성하시오.
    
    입력
    첫째 줄에 100,000보다 작거나 같은 자연수 N이 주어진다.
    
    출력
    첫째 줄부터 N번째 줄 까지 차례대로 출력한다.
    
    예제 입력 1 
    5
    예제 출력 1 
    5
    4
    3
    2
    1
"""


def Q_2742_01():
    n = int(input()) + 1
    for i in range(1, n):
        print(n - i)
    return


def Q_2742_02():
    n = int(input())
    print("\n".join(map(str, range(n, 0, -1))))
    return

Q_2742_01()
Q_2742_02()