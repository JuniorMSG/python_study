"""
    num = int(input())
    test_list = list(map(int, input().split()))
    max_score = max(test_list)

    avg = 0
    for i in test_list:
        avg += i / max_score * 100

    print(avg / len(test_list))

"""

"""
문제
정수 n개가 주어졌을 때, n개의 합을 구하는 함수를 작성하시오.

작성해야 하는 함수는 다음과 같다.

Python 2, Python 3, PyPy, PyPy3: def solve(a: list) -> int
a: 합을 구해야 하는 정수 n개가 저장되어 있는 리스트 (0 ≤ a[i] ≤ 1,000,000, 1 ≤ n ≤ 3,000,000)
리턴값: a에 포함되어 있는 정수 n개의 합 (정수)
"""
def Q_15596(a):
    return sum(a)

Q_15596([1,2,3])