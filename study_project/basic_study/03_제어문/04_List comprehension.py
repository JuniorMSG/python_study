"""
Author  : SG
DATE    : 2021-05-27

03. 제어문 , 반복문

03_01. if       (제어문)
03_02. while    (반복문)
03_03. for      (반복문)
★ 03_04. List comprehension   (리스트 내포)
03_04. Question algorithm

★ 03_04. 리스트 내포(List comprehension)
    01. 리스트 내포(List comprehension)란?
    02. 리스트 내포 사용방법

"""

"""
    01. 리스트 내포(List comprehension)란?
    리스트 내부에 for문을 포함하는 구문을 말한다.
        1. 편리하고 직관적인 프로그램을 만들 수 있다.
"""

"""
    02. 리스트 내포 사용방법 
        02_01. 기본 구문
            표현식 for 항목 in 반복가능 객체 (if 조건문)
        02_02. 2중 for문 
"""

print("\n", "=" * 5, "02. 리스트 내포 사용방법 ", "=" * 5)
print("\n", "=" * 3, "02_01. 기본 구문", "=" * 3)

num = [1, 2, 3, 4, 5]

# 표현식 for 항목 in 반복가능 객체 (if 조건문)
num_lst_com_01 = [num*2 for num in range(1, 6)]
num_lst_com_02 = [num for num in range(1, 6) if num % 2 == 0]

print(num)
print(num_lst_com_01)
print(num_lst_com_02)


print("\n", "=" * 3, "02_02. 2중 for문 ", "=" * 3)

# num_lst_com_0201 = [x % y ]
# print(num_lst_com_0201)

num_lst_com_0201 = [x*y for x in range(1, 10) for y in range(1, 10)]
print(num_lst_com_0201)