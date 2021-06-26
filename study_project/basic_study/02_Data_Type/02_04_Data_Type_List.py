"""
Author  : SG
DATE    : 2021-05-18
02_Data_Type

02_01. Integer      (숫자형)
02_02. String       (문자열)
02_03. Boolean      (참&거짓 자료형)
★ 02_04. List      (리스트)
02_05. Tuple        (수정 불가능 리스트)
02_06. Dictionary   (HashMap - Key- Map Type)
02_07. Set          (집합 자료형)
02_08. Variable     (변수)

★ 02_04. List      (리스트)
    01. List 자료형 이란?
    02. List 자료형 만드는 방법
    03. List 인덱싱, 슬라이싱
    04. List 연산하기
    05. 값 수정, 삭제
    06. 내장함수(built-in Function)
"""

"""
01. List 자료형 이란?
    01. List 자료형이란 
        - 자료를 순서대로 저장하는 자료구조 
        - 여러 자료가 일직선으로 연결된 선형 구조
        - 맨앞 Head, 맨뒤 Tail 이라고 부름
"""


"""
02. List 자료형 만드는 방법
    [] 형태로 선언하면 된다.

"""
print("=" * 5, "02. List 자료형 만드는 방법", "=" * 5)
list_int = [1, 2, 3, 4]
list_str = ['my', 'life']
list_inner_list = [list_int, list_str]

print("list_int         :", list_int)
print("list_str         :", list_str)
print("list_inner_list  :", list_inner_list)


"""
03. List 인덱싱, 슬라이싱
    문자열 처럼 똑같이 앞에부터 0으로 시작된다
    리스트 안에 리스트가 있을경우 [][] 이중 배열형태로 사용하면 된다.
    그 안에 또 리스트가 있다면 [][][] 삼중 배열형태로 사용하면 된다.
"""
print("=" * 5, "03. List 인덱싱, 슬라이싱", "=" * 5)
list_int = [1, 2, 3, 4]
list_str = ['my', 'life']
list_2inner_list = [list_int, list_str]
list_3inner_list = [list_int, list_2inner_list]

print("list_int[0]                  :", list_int[0])
print("list_str[0]                  :", list_str[0:2])
print("list_inner_list[0]           :", list_inner_list[0])
print("list_inner_list[1]           :", list_2inner_list[1])
print("list_3inner_list             :", list_3inner_list)
print("list_3inner_list[1][1][1]    :", list_3inner_list[1][1][1])



"""
04. List 연산하기
    문자열 처럼 + , * 사용이 가능하다.
    01. +
    02. *
    03. 길이 구하기 (len)
"""
print("=" * 5, "04. List 연산하기", "=" * 5)

lst_a = [1,2,3]
lst_b = [3,4,5]
lst_sum = lst_a + lst_b
lst_mul = lst_a * 3
print("01. lst_sum      : ", lst_sum)
print("02. lst_mul      : ", lst_mul)
print("03. len(lst_mul) : ", len(lst_mul))



"""
05. 값 수정, 삭제
    01. 수정
    02. 삭제 (del, remove, pop)
        del     : 특정 요소를 지목하여 삭제
        remove  : 리스트에서 첫 번째로 나오는 x삭제
        pop     : 맨 마지막 요소를 돌려주고 그 요소는 삭제한다.
"""
print("=" * 5, "05. 값 수정, 삭제", "=" * 5)
lst_a = [1,2,3]
lst_b = [3,4,5]
lst_a[0] = 999
lst_a[1] = lst_b
print("01. 수정:", lst_a)

lst_a = [1,2,3]
lst_a = lst_a * 3
print("lst_a :", lst_a)

del lst_a[2]
print("02. 삭제 : del", lst_a)
lst_a.remove(1)
print("02. 삭제 : remove :", lst_a)
lst_a.pop()
print("02. 삭제 : pop :", lst_a)

"""
06. List 내장함수(built-in Function)
    1. 추가
        append(x) : x를 리스트 요소로 추가
        insert(x,y) : x번째 위치에 y를 삽입한다.
        extend(lst) : a 리스트에 lst 리스트를 더한다 (리스트만 올 수 있다)
        
    2. 삭제
        remove(x) : 리스트에서 첫 번째로 나오는 x삭제
        pop() : 맨 마지막 요소를 돌려주고 그 요소는 삭제한다.

    3. 정렬
        sort() : 요소 정렬
        reverse() : 리스트를 역순으로 뒤집는다.
        
    4. 기타
        index(x) : x값이 있으면 위치 값을 돌려준다. 없을시 에러 발생
        count(x) : 리스트에 포함된 x의 개수 세기
"""


print("=" * 5, "06. List 내장함수(built-in Function)", "=" * 5)
lst_a = [1,2,3]
lst_a.append(6)

print("=" * 5, "06_01. 추가", "=" * 5)
print("lst_a.append(6)      :", lst_a)
lst_a.insert(3, 9)
print("lst_a.insert(3, 9)   :", lst_a)
lst_a.extend(lst_a)
print("lst_a.extend(lst_a)  :", lst_a)


print("=" * 5, "06_02. 삭제", "=" * 5)

lst_a.remove(3)
print("lst_a.remove(3)      :", lst_a)
lst_a.pop()
print("lst_a.pop()          :", lst_a)


print("=" * 5, "06_03. 정렬", "=" * 5)
lst_a.sort()
print("lst_a.sort()         :", lst_a)
lst_a.reverse()
print("lst_a.reverse()      :", lst_a)


print("=" * 5, "06_04. 기타", "=" * 5)

print("lst_a.index(3)       :", lst_a.index(3))
print("lst_a.index(4)       : ERROR")
print("lst_a.count(3)       :", lst_a.count(3))
print("lst_a.count(2)       :", lst_a.count(2))
