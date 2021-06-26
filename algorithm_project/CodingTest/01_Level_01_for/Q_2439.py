"""
    https://www.acmicpc.net/step/3
    Subject : Coding Test For
"""

"""
    https://www.acmicpc.net/problem/2439
    def Q_2439():
    첫째 줄에는 별 1개, 둘째 줄에는 별 2개, N번째 줄에는 별 N개를 찍는 문제
    
    하지만, 오른쪽을 기준으로 정렬한 별(예제 참고)을 출력하시오.
    
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

def Q_2439():
    n = int(input()) + 1
    b = []
    for i in range(1, n, 1):
        print(("*" * i).rjust(n - 1, " "))
    return

Q_2439()
