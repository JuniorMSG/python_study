"""
Author  : SG
DATE    : 2021-05-22
02_Data_Type

02_01. Integer      (숫자형)
02_02. String       (문자열)
02_03. Boolean      (참&거짓 자료형)
02_04. List         (리스트)
02_05. Tuple        (수정 불가능 리스트)
02_06. Dictionary   (HashMap - Key- Map Type)
02_07. Set          (집합 자료형)
★ 02_08. Variable  (변수)

★ 02_08. Variable  (변수)
    01. 변수란?
    02. 변수 선언 방법
    03. 변수의 참조
    04. 변수의 복사
    05. 다양한 변수 선언 방법
"""

"""
01. 변수란?
    파이썬의 모든것은 객체로 이루어져 있으며 객체는 속성값 (attributes or value) 행동 (behavior)을 가지고 있습니다. 
    파이썬에서 사용하는 변수는 객체를 가르킵니다.

    변수란 정보를 저장하는 메모리 공간에 이름을 붙인것을 뜻합니다.
    그 메모리 공간은 정보를 저장하는 용도로 만들어지며 그 공간을 찾기 쉽게 만들기 위해서 이름을 지정하는걸
    변수를 선언한다고 합니다.

    그 변수에 특정 데이터를 집어넣으면 그것을 변수에 값을 할당한다고 합니다.
"""


"""
02. 변수 선언 방법
    파이썬은 JavaScript랑 비슷하게 데이터 타입을 미리 선언하지 않고 변수를 선언합니다.
"""
print("\n", "=" * 5, "02. 변수 선언 방법", "=" * 5)

var_01 = 1
var_02 = [1,2]
var_03 = (1,2,3)

print("id(var_01) :", id(var_01))
print("id(var_02) :", id(var_02))
print("id(var_03) :", id(var_03))



"""
03. 변수의 참조
    변수에서 참조 한다는 것은 메모리 위치를 찾아 간다는 의미로 생각하면 됩니다.
        1. 하나만 수정해도 전체가 변경된다.

        2. A변수를 B에 할당하면 A와 B가 참조하는 메모리 위치가 같아진다.
           데이터에 대한 수정시 A, B가 동시에 변하지만 A에 특정 값을 재할당 한다면
           A와 B는 달라지게 된다. 

        3. B는 A를 참조하는게 아닌 A가 가지고 있던 메모리 위치를 가지고 있을 뿐이다.
        (리스트 , Set, Dictionary)
"""
print("\n", "=" * 5, "03. 변수의 참조", "=" * 5)

var_01 = 1
var_02 = [1,2,3]
var_03 = (1,2,3)
var_dic_01 = {"name":1 ,"data":2, "Level":3}

var_str_01 = "Hello"
var_str_02 = var_str_01

# 참조 예시
print("\n", "=" * 5, "03_01. 참조 리스트 예시", "=" * 5)
var_04 = var_02
print("var_02, var_04           :", var_02, var_04)
print("id(var_02), id(var_04)   :", id(var_02), id(var_04))
print("var_02 is var_04         :", var_02 is var_04)

var_04[2] = 8 
print("var_02, var_04           :", var_02, var_04)
print("id(var_02), id(var_04)   :", id(var_02), id(var_04))
print("var_02 is var_04         :", var_02 is var_04)

# var_04는 var_02를 참조하는게 아니라 var_02의 변수의 위치를 참조한다.
var_02 = [9,9,9]
print("var_02, var_04           :", var_02, var_04)
print("id(var_02), id(var_04)   :", id(var_02), id(var_04))
print("var_02 is var_04         :", var_02 is var_04)


print("\n", "=" * 5, "03_02. 참조 Dictionary 예시", "=" * 5)
var_dic_02 = var_dic_01
print("var_dic_01, var_dic_02           :", var_dic_01, var_dic_02)
print("id(var_dic_01), id(var_dic_02)   :", id(var_dic_01), id(var_dic_02))
print("var_dic_01 is var_dic_02         :", var_dic_01 is var_dic_02)

var_dic_02["name"] = "name"
print("var_dic_01, var_dic_02           :", var_dic_01, var_dic_02)
print("id(var_dic_01), id(var_dic_02)   :", id(var_dic_01), id(var_dic_02))
print("var_dic_01 is var_dic_02         :", var_dic_01 is var_dic_02)


# var_dic_02는 var_dic_01을 참조하는게 아니라 var_dic_01의 변수의 위치를 참조한다.
var_dic_01 = {"NEW" : "new_dictinary"}
print("var_dic_01, var_dic_02           :", var_dic_01, var_dic_02)
print("id(var_dic_01), id(var_dic_02)   :", id(var_dic_01), id(var_dic_02))
print("var_dic_01 is var_dic_02         :", var_dic_01 is var_dic_02)


print("\n", "=" * 5, "03_03. 문자열 & 숫자", "=" * 5)

print("변경전 : var_str_01 is var_str_02 : ", var_str_01 is var_str_02)
var_str_02 = "Hello2"
print("변경후 : var_str_01 is var_str_02 : ", var_str_01 is var_str_02)
var_str_02 = "Hello"
print("같은값 : var_str_01 is var_str_02 : ", var_str_01 is var_str_02)

# 숫자 변수 선언
var_int_01 = 5
var_int_02 = var_int_01

print("변경전 : var_int_01 is var_int_02 : ", var_int_01 is var_int_02)
var_int_02 = 9
print("변경전 : var_int_01 is var_int_02 : ", var_int_01 is var_int_02)
var_int_02 = 5
print("같은값 : var_int_01 is var_int_02 : ", var_int_01 is var_int_02)




"""
04. 변수의 복사
    리스트 등에서 같은 값을 가지면서 다른 메모리 위치를 가지도록 만들때 복사를 합니다.
    수정시 다른 객체에 영향을 주지 않습니다.

    1. [:]
    2. copy 모듈 
"""
print("\n", "=" * 5, "04. 변수의 복사", "=" * 5)

print("\n", "=" * 5, "04_01. [:]", "=" * 5)
var_copy_01 = [1,2,3]
var_copy_02 = var_copy_01[:]

print("var_copy_01 , var_copy_02        :", var_copy_01 , var_copy_02)
print("var_copy_01 is var_copy_02       :", var_copy_01 is var_copy_02)
print("id(var_copy_01) id(var_copy_02)  :", id(var_copy_01), id(var_copy_02))


print("\n", "=" * 5, "04_02. copy 모듈 ", "=" * 5)
from copy import copy

var_copy_01 = [1,2,3]
var_copy_02 = var_copy_01.copy()
print("var_copy_01 , var_copy_02        :", var_copy_01 , var_copy_02)
print("var_copy_01 is var_copy_02       :", var_copy_01 is var_copy_02)
print("id(var_copy_01) id(var_copy_02)  :", id(var_copy_01), id(var_copy_02))


"""
05. 다양한 변수 선언 방법
    파이썬에서만 존재하는(?) 특이한 선언 방식이 있습니다.
    함수에서도 리턴값이 2개가 될 수 있으며
    변수를 스위칭할때 임시변수 선언을 할 필요도 없습니다.

    1. 복수개의 변수에 한번에 할당하기
    2. 변수 스위칭하기
    3. 한번에 할당하기
"""
print("\n", "=" * 5, "05. 다양한 변수 선언 방법", "=" * 5)

print("\n", "=" * 5, "05_01. 2개의 변수에 한번에 할당하기", "=" * 5)

a, b, c = ('Hello', 'Python', 'Project')
print("a, b, c  :", a, b, c)

(a, b, c) = 'Hello', 'Python', 'Project'
print("a, b, c  :", a, b, c)

[a, b, c] = ['Hello', 'Python', 'Project']
print("a, b, c  :", a, b, c)

print("\n", "=" * 5, "05_02. 변수 스위칭하기", "=" * 5)
a, b, c = c, a, b
print("a, b, c  :", a, b, c)

print("\n", "=" * 5, "05_03. 한번에 할당하기", "=" * 5)
a = b = c = 'Hello'
print("a, b, c  :", a, b, c)

