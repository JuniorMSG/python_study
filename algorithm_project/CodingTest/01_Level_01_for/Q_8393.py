"""
    https://www.acmicpc.net/step/3
    Subject : Coding Test For
"""

"""
    https://www.acmicpc.net/problem/8393
    
    def Q_8393():
    n이 주어졌을 때, 1부터 n까지 합을 구하는 프로그램을 작성하시오.
    
    입력
    첫째 줄에 n (1 ≤ n ≤ 10,000)이 주어진다.
    
    출력
    1부터 n까지 합을 출력한다.
    
    예제 입력 1 
    3
    예제 출력 1 
    6
"""

def Q_8393_01():
    print(sum(list(range(0, int(input()) + 1))))
    return

def Q_8393_02():
    n = int(input())
    print((n ** 2 + n) // 2)
    return

Q_8393_01()
Q_8393_02()