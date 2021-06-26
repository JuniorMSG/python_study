import time
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