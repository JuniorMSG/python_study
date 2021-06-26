
"""
    03. 예제
        03_01. 백준 2739번 문제
        ★ 03_02. 백준 10950번 문제
"""

"""
def Q_10950():
백준 10950번 문제
두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

입력
첫째 줄에 테스트 케이스의 개수 T가 주어진다.
각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)

출력
각 테스트 케이스마다 A+B를 출력한다.
"""
def Q_10950():
    testcase = int(input("입력받을 케이스 개수 : "))
    listCase = []

    for i in range(0, testcase):
        listCase.append(input("두 정수를 입력하세요 : "))

    for i, item in enumerate(listCase):
        num = item.split()
        print(int(num[0]) + int(num[1]))
    return


print("\n", "=" * 3, "03_02. 백준 10950번 문제", "=" * 3)
Q_10950()