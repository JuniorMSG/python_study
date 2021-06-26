"""
    순환 호출 (circular call) - 재귀적 호출 (recursive call)
        함수 내부에서 함수가 자기 자신을 또다시 호출하는 행위를 의미합니다.
        이러한 재귀 호출은 자기가 자신을 계속해서 호출하므로, 끝없이 반복되게 됩니다.

        순환 호출의 단점은 알고리즘 특성에 따라 다르겠지만
        수행시간이 오래 걸린다.

    꼬리재귀  (Tail Recursion)
        재귀 호출이 끝난 후 현재 함수에서 추가 연산을 요구하지 않도록 구현하는 재귀의 형태
        함수 호출이 반복되어 스택이 깊어지는 문제를 컴파일러가 선형으로 처리 하도록 알고리즘을 바꿔 스택을 재사용할 수 있게 됩니다.
        꼬리 재귀를 사용하기 위해서는 컴파일러가 이런 최적화 기능을 지원해야 합니다.
"""

"""
    01. 예시 피보나치 수열
        01_01. 재귀함수 구현
        01_02. 꼬리 재귀로 구현
"""

print("\n", "=" * 5, "01. 예시 피보나치 수열 ", "=" * 5)
import time


def fibonacci_numbers(num):
    if num <= 1:
        return num
    else:
        return fibonacci_numbers(num-1) + fibonacci_numbers(num-2)


print("\n", "=" * 3, "01_01. 재귀함수 구현", "=" * 3)
start = time.time()  # 시작 시간 저장
print('실행값 :', fibonacci_numbers(30))
print('실행시간 측정 : ', time.time() - start)  # 실행시간 측정


print("\n", "=" * 3, "01_02. 꼬리 재귀로 구현", "=" * 3)


def fibonacci_numbers_tail(num, first_fibo, second_fibo):
    if num <= 1:
        return num * first_fibo
    else:
        return fibonacci_numbers_tail(num-1, first_fibo + second_fibo, first_fibo)

import sys
print(sys.setrecursionlimit)
sys.setrecursionlimit(10**7)
start = time.time()  # 시작 시간 저장
print(start)
print('실행값 : ', fibonacci_numbers_tail(1000, 1, 0))  # 실행시간 측정
print('실행시간 측정 : ', time.time() - start)  # 실행시간 측정


