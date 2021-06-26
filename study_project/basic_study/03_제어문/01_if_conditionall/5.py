"""
05. 조건문이 복수일때 실행 순서와 생략

조건문은 조건이 만족하느냐 만족하지 않느냐를 따진다.

return이 비어있거나 없는경우는 False로 떨어진다. EX) print("1234") = false

대표적으로 and문은 조건이 하나라도 False면 실행이 안되기 때문에 뒤의 조건을 보지 않는다.
or문은 조건이 하나라도 True이면 실행이 되기 때문에 뒤의 조건을 보지 않는다.
    01. and : 하나라도 False가 나오면 뒤의 문장을 실행하지 않는다.
    02. or  : 하나라도 True가 나오면 뒤의 문장을 실행하지 않는다.

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
