"""
    https://www.acmicpc.net/step/3
    Subject : Coding Test For
"""

"""
    https://www.acmicpc.net/problem/2438
    def Q_2438():
    첫째 줄에는 별 1개, 둘째 줄에는 별 2개, N번째 줄에는 별 N개를 찍는 문제
    
    입력
    첫째 줄에 N(1 ≤ N ≤ 100)이 주어진다.
    
    출력
    첫째 줄부터 N번째 줄까지 차례대로 별을 출력한다.
    
    예제 입력 1 
    5
    예제 출력 1 
    *
    **
    ***
    ****
    *****
"""


def Q_2438():
    a = int(input())
    b = ""
    for i in range(1, a+1):
        print("*" * i)
    return

Q_2438()