"""
Author  : SG
DATE    : 2021-05-18
02_Data_Type

02_01. Integer      (숫자형)
02_02. String       (문자열)
★ 02_03. Boolean   (참&거짓 자료형)
02_04. List         (리스트)
02_05. Tuple        (수정 불가능 리스트)
02_06. Dictionary   (HashMap - Key- Map Type)
02_07. Set          (집합 자료형)
02_08. Variable     (변수)

★ 02_03. Boolean   (참&거짓 자료형)
    01. Boolean 참&거짓 자료형 이란?
    02. 자료형별 참 거짓 기준, 테스트
"""

"""
01. Boolean 참&거짓 자료형 이란?
    01. True , False로만 구성된 자료형을 뜻한다.
"""


"""
02. 자료형별 참 거짓 기준 및 연산 테스트
    01. 문자열  - 값 O - True, X = False
    02. List, Tuple, Dic, Set
        - 갯수 1개 이상 - True , 0개 - Fals
    03. 숫자 - 0 거짓 , 나머지 참
    04. None - 거짓
"""
print("=" * 5, "02. 자료형별 참 거짓", "=" * 5)
print("=" * 5, "02-01. 문자열", "=" * 5)
print('bool("python") : ', bool("python"))
print('bool('') : ', bool(''))

list_val_n  = []
tuple_val_n = ()
dic_val_n = {}
set_val_n = set(list_val_n)

list_val_y  = [1, 1, 2, 3]
tuple_val_y = (1, 1, 2, 3)
dic_val_y = {"val":1}
set_val_y = set(list_val_y)

print("=" * 5, "02-02. List, Tuple, Dic, Set", "=" * 5)
print('bool(list_val_n)     : ' , bool(list_val_n))
print('bool(tuple_val_n)    : ' , bool(tuple_val_n))
print('bool(dic_val_n)      : ' , bool(dic_val_n))
print('bool(set_val_n)      : ' , bool(set_val_n))

print('bool(list_val_y)     : ' , bool(list_val_y))
print('bool(tuple_val_y)    : ' , bool(tuple_val_y))
print('bool(dic_val_y)      : ' , bool(dic_val_y))
print('bool(set_val_y)      : ' , bool(set_val_y))

print("=" * 5, "02-03. 숫자 - 0 거짓 , 나머지 참", "=" * 5)
print('bool(1)      : ' , bool(1))
print('bool(152)    : ' , bool(152))
print('bool(1.33)   : ' , bool(1.33))
print('bool(-1.33)  : ' , bool(-1.33))
print('bool(0)      : ' , bool(0))

print("=" * 5, "02-04. None - 거짓", "=" * 5)
print('None      : ' , bool(None))