"""
    https://www.acmicpc.net/step/3
    Subject : Coding Test For
"""

"""
    https://www.acmicpc.net/problem/2741
    Q_2741():
    자연수 N이 주어졌을 때, 1부터 N까지 한 줄에 하나씩 출력하는 프로그램을 작성하시오.
    
    입력
    첫째 줄에 100,000보다 작거나 같은 자연수 N이 주어진다.
    
    출력
    첫째 줄부터 N번째 줄 까지 차례대로 출력한다.
    
    예제 입력 1 
    5
    예제 출력 1 
    1
    2
    3
    4
    5
"""
def Q_2741():
    for i in range(1, int(input()) + 1):
        print(i)
    return

Q_2741()