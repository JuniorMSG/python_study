"""
Author  : SG
DATE    : 2021-05-27

03_ 제어문 , 반복문

★ 03_01. if (제어문)
03_02. while
03_03. for
03_04. Question algorithm

★ 03_01. if (제어문)
    01. if 제어문 이란?
    02. 기본 구조
    03. 연산자
        03-01. 비교, 관계 연산자
        03-02. and or not
        03-03. in, not in
    04. pass
    05. 조건문이 복수일때 실행 순서와 생략

"""

"""
01. if 제어문 이란?

제어 흐름(control flow)을 나타내는 구문을 뜻한다.
실행 방식에 따라서 분기, 조건 분기, Loop, 서브루틴, 중지등으로 구분된다.
if 제어문은 분기, 조건 분기에 관련된 제어문이다.

파이썬의 특이한점
    1. else if 대신 elif를 사용한다.
    2. switch 구문 자체가 존재하지 않는다.
    3. 괄호를 사용하지 않는다
"""


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


"""
03. 연산자
    03-01. 비교, 관계 연산자
    03-02. 논리 연산자
    03-03. 포함 연산
"""

print("\n", "=" * 5, "03. 연산자", "=" * 5)

print("\n", "=" * 3, "03_01. 비교, 논리 연산자", "=" * 3)
print("3 > 5  : x가 y보다 크다", 3 > 5)
print("3 < 5  : x가 y보다 작다", 3 < 5)
print("3 >= 5 : x가 y보다 크거나 같다.", 3 >= 5)
print("3 <= 5 : x가 y보다 작거나 같다.", 3 <= 5)
print("3 == 5 : x와 y가 같다.", 3 == 5)
print("3 != 5 : x와 y가 같지 않다.", 3 != 5)

print("\n", "=" * 3, "03_02. and or not", "=" * 3)
print("x or  y : 둘중 하나만 참이면 참:", True or False)
print("x and y : 둘다 참이여야 참:", True and True)
print("not x : x가 거짓이면 참이다.:", not False)


print("\n", "=" * 3, "03_03. in, not in", "=" * 3)
data = [1, 2, 3]
print("3 in data : 3이 데이터 안에 포함된다. ", 3 in data)
print("3 not in data : 3이 데이터 안에 포함되지 않는다. ", 3 not in data)

"""
04. pass
파이썬은 조건문 내부에 아무것도 작성하지 않으면 오류가 나는데
아무런 일도 하지 않도록 설정할때 사용한다.
"""

print("\n", "=" * 5, "04. pass", "=" * 5)

data = [1, 2, 3]
if 3 in data:
    pass
else:
    print(data)

"""
05. 조건문이 복수일때 실행 순서와 생략

조건문은 조건이 만족하느냐 만족하지 않느냐를 따진다.

대표적으로 and문은 조건이 하나라도 False면 실행이 안되기 때문에 뒤의 조건을 보지 않는다.
or문은 조건이 하나라도 True이면 실행이 되기 때문에 뒤의 조건을 보지 않는다.
    01. and : 하나라도 False가 나오면 뒤의 문장을 실행하지 않는다.
    02. or  : 하나라도 True가 나오면 뒤의 문장을 실행하지 않는다.
    03. return이 비어있거나 없는경우는 False로 떨어진다. EX) print("1234") = false

"""

print("\n", "=" * 5, "05. 조건문이 복수일때 실행 순서와 생략", "=" * 5)

print("\n", "=" * 3, "05_01. and는 False가 나오면 뒤의 문장을 실행하지 않는다.", "=" * 3)
if True and print("1. AND 조건절 실행됨"):
    print("===" * 5)
    print("1. AND 테스트")

if False and print("1. AND 조건절 실행안됨"):
    print("===" * 5)
    print("1. AND 테스트")

print("\n", "=" * 3, "05_02. or은 True가 나오면 뒤의 문장을 실행하지 않는다.", "=" * 3)
if True or print("OR 조건절 실행안됨"):
    print("===" * 5)
    print("or테스트")

if False or print("OR 조건절 실행됨"):
    print("===" * 5)
    print("or테스트")

print("\n", "=" * 3, "05_03. return이 비어있거나 없는경우는 False로 떨어진다.", "=" * 3)

# 객체 리턴값이 True조건인 경우 True가 되긴하는데
# return이 비어있거나 없는경우 , False조건에 해당하는 경우 (숫자 0, 문자열 공백, 리스트갯수 0개 False로 떨어진다.
if print("프린트 테스트"):
    print("프린트 테스트 : ", True)
else:
    print("프린트 테스트 : ", False)

if len([1,2,3]):
    print("len([1,2,3]) : ", True)
else:
    print("len([1,2,3]) : ", False)

if len([]):
    print("len([]) : ", True)
else:
    print("len([]) : ", False)

def ret_lst():
    return []

if ret_lst():
    print("ret_lst() : ", True)
else:
    print("ret_lst() : ", False)






