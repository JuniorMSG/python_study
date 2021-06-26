"""
 2557 : Hello World
"""
def Q_2557():
    print("Hello World!")

"""
    10718 : 주어진 예제처럼 출력하기
    강한친구 대한육군
    강한친구 대한육군
"""
def Q_10718():
    print("강한친구 대한육군")
    print("강한친구 대한육군")

"""
    10171 : 고양이를 출력한다
\    /\
 )  ( ')
(  /  )
 \(__)|
"""
def Q_10171():
    print("\\    /\\")
    print(" )  ( ')")
    print("(  /  )")
    print(" \\(__)|")
    return

def Q_10172():
    print("|\_/|")
    print("|q p|   /}")
    print("( 0 )\"\"\"\\")
    print("|\"^\"`   |")
    print("||_/=\\\__|")
    return

def Q_1000():
    a, b = input().split()
    print(int(a) + int(b))
    return

def Q_1001():
    a, b = input().split()
    print(int(a) - int(b))
    return

def Q_10998():
    a, b = input().split()
    print(int(a) * int(b))
    return

def Q_1008():
    a, b = input().split()
    print(int(a) / int(b))
    return

def Q_10869():
    a, b = input().split()
    a = int(a)
    b = int(b)
    print(int(a+b))
    print(int(a-b))
    print(int(a*b))
    print(int(a/b))
    print(int(a) % int(b))
    return

"""
Q_10430
(A+B)%C는 ((A%C) + (B%C))%C 와 같을까?
(A×B)%C는 ((A%C) × (B%C))%C 와 같을까?
세 수 A, B, C가 주어졌을 때, 위의 네 가지 값을 구하는 프로그램을 작성하시오.

입력
첫째 줄에 A, B, C가 순서대로 주어진다. (2 ≤ A, B, C ≤ 10000)

출력
첫째 줄에 (A+B)%C, 둘째 줄에 ((A%C) + (B%C))%C, 셋째 줄에 (A×B)%C, 넷째 줄에 ((A%C) × (B%C))%C를 출력한다.
"""

def Q_10430():
    a, b, c= input().split()
    a, b, c = int(a), int(b), int(c)
    print((a+b)%c)
    print(((a%c) + (b%c))%c)
    print((a*b)%c)
    print(((a%c) * (b%c))%c)
    return


def Q_2588():
    """
    문제
    (세 자리 수) × (세 자리 수)는 다음과 같은 과정을 통하여 이루어진다.
     (1)과 (2)위치에 들어갈 세 자리 자연수가 주어질 때 (3), (4), (5), (6)위치에 들어갈 값을 구하는 프로그램을 작성하시오.

    입력
    첫째 줄에 (1)의 위치에 들어갈 세 자리 자연수가, 둘째 줄에 (2)의 위치에 들어갈 세자리 자연수가 주어진다.

    출력
    첫째 줄부터 넷째 줄까지 차례대로 (3), (4), (5), (6)에 들어갈 값을 출력한다.

    :return:
    """

    a = input()
    b = input()
    a, b = int(a), int(b)
    print(a*(b%10))
    print(int(a*(b%100 - b%10)/10))
    print(int(a*(b%1000 - b%100)/100))
    print(int(a*b))





