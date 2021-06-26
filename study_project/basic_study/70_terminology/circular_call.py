"""
    순환 호출 (circular call) - 재귀적 호출 (recursive call)
        함수 내부에서 함수가 자기 자신을 또다시 호출하는 행위를 의미합니다.
        이러한 재귀 호출은 자기가 자신을 계속해서 호출하므로, 끝없이 반복되게 됩니다.

    순환 호출의 단점은 알고리즘 특성에 따라 다르겠지만
    수행시간이 오래 걸린다.
"""

"""
    01. 예시 피보나치 수열
        01_01. 재귀함수 구현
        01_02. 반복문 구현
        01_03. 꼬리 재귀로 구현
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


print("\n", "=" * 3, "01_02. 반복문 구현", "=" * 3)
fibo_save = 0
fibo_first = 1
fibo_second = 0

start = time.time()  # 시작 시간 저장
for cnt in range(1, 1000):
    fibo_save = fibo_first + fibo_second
    fibo_second, fibo_first = fibo_first, fibo_save
print('실행값 :', fibo_save)
print('실행시간 측정 :', time.time() - start)  # 실행시간 측정

print("\n", "=" * 3, "01_03. 꼬리 재귀로 구현", "=" * 3)


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


