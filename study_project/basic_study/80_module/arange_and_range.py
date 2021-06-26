""" NW
    arange, range 
    순서대로 리스트에 값을 생성하는 내장함수
 

    n dimension array (NDarray)  :
    1D array    :   x축 존재
    2D array    :   x, y축 존재
    3D array    :   x, y, z축 존재

    01. 기본 사용법
    02. array에서의 data 타입
    03. 인덱싱 , 슬라이싱
    04. Fancy 인덱싱
    05. Bollean 인덱싱
    06. arange와 range
"""

"""
    01. 기본 사용법
import numpy as np

"""
import numpy as np

print("\n", "=" * 5, "01. 기본 사용법 ", "=" * 5)
arr1 = np.array([1,2,3,4], dtype=int)
arr2 = np.array([[1,2,3,4], [5,6,7,8]])
print('arr1 :', arr1)
print('arr2 :', arr2)
print('arr1.shape :', arr1.shape)
print('arr2.shape :', arr2.shape)

"""
   02. array에서의 data 타입
        하나의 단일 데이터 타입만 허용된다.
        01. case1. int + float 
            자동으로 정수가 실수형으로 변환
        02. case2. int + float  dtype 지정시
            지정된 데이터 타입으로 변환
        03. case3. int + str    
            자동으로 문자열 타입으로 변환
        04. case3. int + str    dtype 지정시    
            변환 가능한 dtype으로 변환
            변환이 불가능한 경우 변환되지 않는다.
"""

print("\n", "=" * 5, "02. array에서의 data 타입", "=" * 5)
print("\n", "=" * 3, "02_01. int + float ", "=" * 3)
print(np.array([1, 2, 3.14, 4, 5.5]))

print("\n", "=" * 3, "02_02. int + float  dtype 지정시", "=" * 3)
print(np.array([1, 2, 3.14, 4, 5.5], dtype=int))

print("\n", "=" * 3, "02_03. int + str    ", "=" * 3)
arr3 = np.array([1, 2, '3.14', '문자열',4, 5.5])
print(arr3)
print("문자열 취급받아서 연산된다. : ", arr3[0] + arr3[1])

print("\n", "=" * 3, "02_04. int + str    dtype 지정시", "=" * 3)
print("변환이 불가능한 경우 변환되지 않는다.")
# invalid literal for int() with base 10: '3.14' 에러 발생
# print(np.array([1, 2, '3.14', '문자열',4, 5.5], dtype=int))


"""
   03. 인덱싱 슬라이싱
        차원별로 인덱싱을 지정한다.
        01. 1차원
            기본 리스트 인덱싱과 동일하다.
            리스트 글 참조
        02. 2차원
            lst(x, y) 형태로 구성된다.
            x = 행, y는 열이 된다.

"""

print("\n", "=" * 5, "03. array에서의 data 타입", "=" * 5)
print("\n", "=" * 3, "03_01. 1차원 ", "=" * 3)

lst1D = np.array([1,2,3,4])
print('lst1D :', lst1D[:-1])

print("\n", "=" * 3, "03_02. 2차원 ", "=" * 3)

lst2D = np.array( [[1,2,3,4] , [5,6,7,8], [9,10,11,12]])
print('lst2D[0, :] :', lst2D[0, :])
print('lst2D[:, 0] :', lst2D[:, 0])
print('lst2D[0:, :2] :', lst2D[0:, :2])
print('lst2D[0:, :2] :', lst2D[2:, :])



"""
   04. Fancy 인덱싱
        특정 index의 집합을 추출하고 싶을때 사용한다.
        01. 1차원

        02. 2차원
            lst(x, y) 형태로 구성된다.
            x = 행, y는 열이 된다.

"""

print("\n", "=" * 5, "04. Fancy 인덱싱", "=" * 5)
print("\n", "=" * 3, "04_01. 1차원 ", "=" * 3)

idx = [1, 3, 5]
lst1D = np.array([1,22,333,444,5555,66666,77777])
print('lst1D[idx]', lst1D[idx])


print("\n", "=" * 3, "04_02. 2차원 ", "=" * 3)

lst2D = np.array( [[1,2,3,4] , [5,6,7,8], [9,10,11,12]])
print('lst2D[[0,1], :] :', lst2D[[0,1], :])


"""
   05. Boolean 인덱싱
        조건 필터링을 통하여 True, False로 색인한다.
        01. 기본 사용법
        02. 조건절 사용법
            lst[조건필터]

"""

print("\n", "=" * 5, "05. Boolean 인덱싱", "=" * 5)
print("\n", "=" * 3, "05_01. 기본 사용법 ", "=" * 3)

lst1D = np.array([1, 2, 3, 4, 5, 6, 7, 8])
idx = [True, False, False, False, True, True, True, True]

# 두 리스트간의 차원이 맞지 않을경우 아래와 같은 에러가 발생한다.
#boolean index did not match indexed array along dimension 0; dimension is 8 but corresponding boolean dimension is 7
print('Boolean 인덱싱 : ', lst1D[idx])

print("\n", "=" * 3, "05_02. 조건절로 인덱싱 ", "=" * 3)

lst1D = np.array([1, 4, 8, 12, 16, 20, 24, 28])
print('조건절 인덱싱 lst1D[lst1D>2] : ', lst1D[lst1D>2])
