"""
문제
두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

입력
입력은 여러 개의 테스트 케이스로 이루어져 있다.

각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)

입력의 마지막에는 0 두 개가 들어온다.

출력
각 테스트 케이스마다 A+B를 출력한다.

예제 입력 1
1 1
2 3
3 4
9 8
5 2
0 0
예제 출력 1
2
5
7
17
7
https://www.acmicpc.net/problem/10952
"""

def Q_10952():
    while True:
        num1, num2 = map(int, input().split())
        if num1 == 0 and num2 == 0:
            break
        else:
            print(num1 + num2)
    return

# while True:
    # num = input().split()
    # print(int(num[0]) + int(num[1]))


"""
문제
두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

입력
입력은 여러 개의 테스트 케이스로 이루어져 있다.

각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)

출력
각 테스트 케이스마다 A+B를 출력한다.

예제 입력 1 
1 1
2 3
3 4
9 8
5 2
예제 출력 1 
2
5
7
17
7
https://www.acmicpc.net/problem/10951
테스트 케이스가 없으므로 try except 사용해주어야함.
"""
def Q_10951():
    while True:
        try:
            num1, num2 = map(int, input().split())
            print(num1 + num2)
        except:
            break
    return


"""
문제
0보다 크거나 같고, 99보다 작거나 같은 정수가 주어질 때 다음과 같은 연산을 할 수 있다. 먼저 주어진 수가 10보다 작다면 앞에 0을 붙여 두 자리 수로 만들고, 각 자리의 숫자를 더한다. 그 다음, 주어진 수의 가장 오른쪽 자리 수와 앞에서 구한 합의 가장 오른쪽 자리 수를 이어 붙이면 새로운 수를 만들 수 있다. 다음 예를 보자.

26부터 시작한다. 2+6 = 8이다. 새로운 수는 68이다. 6+8 = 14이다. 새로운 수는 84이다. 8+4 = 12이다. 새로운 수는 42이다. 4+2 = 6이다. 새로운 수는 26이다.

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
예제 입력 2 
55
예제 출력 2 
3
예제 입력 3 
1
예제 출력 3 
60
예제 입력 4 
0
예제 출력 4 
1

https://www.acmicpc.net/problem/1110
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






