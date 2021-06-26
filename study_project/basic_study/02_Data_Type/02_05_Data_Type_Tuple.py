"""
Author  : SG
DATE    : 2021-05-22
02_Data_Type

02_01. Integer      (숫자형)
02_02. String       (문자열)
02_03. Boolean      (참&거짓 자료형)
02_04. List         (리스트)
★ 02_05. Tuple     (수정 불가능 리스트)
02_06. Dictionary   (HashMap - Key- Map Type)
02_07. Set          (집합 자료형)
02_08. Variable     (변수)

★ 02_05. Tuple (수정 불가능 리스트)
    01. Tuple 자료형 이란?
    02. Tuple 자료형 선언방법
    03. Tuple 인덱싱, 슬라이싱
    04. Tuple 연산
"""

"""
01. Tuple 자료형 이란?
    리스트와 몇가지 점을 제외하고는 거의 비슷하다.
    - 리스트는 []로 튜플은 ()로 둘러싼다.
    - 튜플은 선언 이후 생성, 삭제, 수정이 불가능하다.
"""


"""
02. Tuple 자료형 선언방법
    [] 형태로 선언하면 된다.
"""
print("=" * 5, "02. Tuple 자료형 선언방법", "=" * 5)

tuple1 = ()
tuple2 = ("S",)
tuple3 = ("S", "A", "M")
tuple4 = "S", "A", "M"
tuple5 = (tuple4, tuple3)

print("tuple1   :", tuple1)
print("tuple2   :", tuple2)
print("tuple3   :", tuple3)
print("tuple4   :", tuple4)
print("tuple5   :", tuple5)


"""
03. Tuple 인덱싱, 슬라이싱
    문자열 처럼 똑같이 앞에부터 0으로 시작된다
    리스트 안에 리스트가 있을경우 [][] 이중 배열형태로 사용하면 된다.
    그 안에 또 리스트가 있다면 [][][] 삼중 배열형태로 사용하면 된다.
"""

print("\n", "=" * 5, "03. Tuple 인덱싱, 슬라이싱", "=" * 5)

tuple_indexing = ("S", "A", "M")
print("tuple_indexing :", tuple_indexing[1])

tuple_slicing = 1,2,3,4,5
print("tuple_slicing :", tuple_slicing[2:])


"""
04. Tuple 연산
    +   : 튜플을 더한다
    *   : 튜플은 반복한다
    len : 튜플의 길이를 리턴한다.
"""
print("\n", "=" * 5, "04. Tuple 연산", "=" * 5)

tuple_cal1 = ("S",)
tuple_cal2 = ("S", "A", "M")

print("tuple_cal1 + tuple_cal2 :", tuple_cal1 + tuple_cal2)
print("tuple_cal2 * 3 : ", tuple_cal2 * 3)
print("len(tuple_cal2) : ", len(tuple_cal2))