"""
02. 기본 구조

if 조건문, elif 조건문등 조건문과 함깨 사용한다.
조건문(條件文, conditional)이란 프로그래머가 명시한 불린 자료형 조건이 참인지 거짓인지에 따라
달라지는 계산이나 상황을 수행하는 프로그래밍 언어의 특징이다.

if num >= 90: if 조건문 순서로 작성한다.

기본적으로 if, else, elif 3가지를 사용하여 작성한다.
if문은 조건이 만족하면 문장을 실행하고 끝나기 때문에 상위 if & elif에 포함되는 조건은
다음 elif에 포함할 필요가 없다.

파이썬 언어의 특성상 들여쓰기(indentation)로 구분하는데 다른 언어에선 보통 중괄호 로 구분한다.
"""
print("=" * 5, "02. 기본 구조", "=" * 5)
num = 85
if num >= 90:
    print("A")
elif num >= 80:
    print("B")
elif num >= 70:
    print("C")
elif num >= 60:
    print("D")
else:
    print("F")