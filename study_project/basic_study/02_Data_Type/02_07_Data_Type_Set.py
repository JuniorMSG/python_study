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
★ 02_07. Set       (집합 자료형)
02_08. Variable     (변수)

★ 02_07. Set       (집합 자료형)
    01. Set 자료형 이란?
    02. Set 자료형 선언방법
    03. 교집합, 합집합, 차집합 구하기
    04. 내장함수(built-in Function)
"""

"""
01. Set 자료형 이란?
    Set 자료형은 집합에 관련된걸 처리하기 위한 자료형인데 
    특징으로는 순서가 없으며 (Dictionary 자료형처럼), 중복을 허용하지 않는 특징을 가지고 있다.
"""


"""
02. Set 자료형 선언방법
    Set 자료형은 리스트나 튜플에, 문자열에 Set 키워드를 이용하여 선언헌다.
"""
print("\n", "=" * 5, "02. Set 자료형 선언방법", "=" * 5)

set_02_lst = set([1,2,3,4,1,2,3,4])
set_02_str = set("Hello Python")

print('set_02_lst :', set_02_lst)
print('set_02_str :', set_02_str)



"""
03. 교집합, 합집합, 차집합 구하기
    교집합 : &기호, intersection()
    합집합 : |기호, union()
    차집합 : -기호, difference()
"""
print("\n", "=" * 5, "03. 교집합, 합집합, 차집합 구하기", "=" * 5)

set_03_lst1 = set([1,2,3,4,5,6])
set_03_lst2 = set([4,5,6,7,8,9])

print("\n", "=" * 5, "03_01. 교집합", "=" * 5)
print('set_03_lst1 & set_03_lst2                :', set_03_lst1 & set_03_lst2)
print('set_03_lst1.intersection(set_03_lst2)    :', set_03_lst1.intersection(set_03_lst2))

print("\n", "=" * 5, "03_02. 합집합", "=" * 5)
print('set_03_lst1 | set_03_lst2                :', set_03_lst1 | set_03_lst2)
print('set_03_lst1.union(set_03_lst2)           :', set_03_lst1.union(set_03_lst2))

print("\n", "=" * 5, "03_03. 차집합", "=" * 5)
print('set_03_lst1 - set_03_lst2                :', set_03_lst1 - set_03_lst2)
print('set_03_lst1.difference(set_03_lst2)      :', set_03_lst1.difference(set_03_lst2))


"""
04. 내장함수(built-in Function)
    추가 
        add     : set 자료형에 1개의 값을 추가한다.
        update  : set 자료형에 복수개의 값을 추가한다.
    삭제
        remove  : set 자료형에 특정값을 제거한다 없을경우 에러가 발생한다.
                  예외가 발생했습니다. KeyError 
        discard : set 자료형에 특정값을 제거한다 없을경우 아무일도 일어나지 않는다
    기타 
        pop     : 임의의 요소를 가져온후 제거한다. 요소가 없을경우 에러가 발생한다.
                  에러 : pop from an empty set 
        in      : 요소가 있는지 확인한다.
        len     : 총 길이를 구한다.
        clear   : 모든 요소를 제거한다.
"""

print("\n", "=" * 5, "04. 내장함수(built-in Function)", "=" * 5)

set_04_lst = set([1,2,3,4,5,6])

print("\n", "=" * 5, "04_01. 추가", "=" * 5)

set_04_lst.add(7)
print('set_04_lst.add(7)            :', set_04_lst)

set_04_lst.update([8,9])
print('set_04_lst.update([8,9])     :', set_04_lst)
set_04_lst.update({10,11}) 
print('set_04_lst.update({10,11})   :', set_04_lst)

print("\n", "=" * 5, "04_02. 삭제", "=" * 5)

# set_04_lst.remove(12) => 예외가 발생했습니다. KeyError 
set_04_lst.remove(11) 
print('set_04_lst.remove(11)    :', set_04_lst)
set_04_lst.discard(11) 
print('set_04_lst.discard(11)   :', set_04_lst)


print("\n", "=" * 5, "04_03. 기타", "=" * 5)
print('set_04_lst.pop()     :', set_04_lst.pop())
print('5 in set_04_lst      :', 5 in set_04_lst)
print('11 in set_04_lst     :', 11 in set_04_lst)
print('len(set_04_lst)      :', len(set_04_lst))

set_04_lst.clear()
print('set_04_lst.clear()   :', set_04_lst)

