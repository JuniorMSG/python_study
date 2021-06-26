"""
    subject
        Data analysis module 
        수학 과학 계산용 패키지

    topic
        numpy 모듈

    Describe
        수학 과학 계산용 패키지 

        n dimension array (NDarray)  :
        1D array    :   x축 존재
        2D array    :   x, y축 존재
        3D array    :   x, y, z축 존재

        행렬의 차원 shape에 대한 내용을 이해하고 하면 편한다.

    Contents
        01. 기본 사용법
        02. numpy array에서 data 타입과
        03. 인덱싱 , 슬라이싱
        04. Fancy 인덱싱
        05. Bollean 인덱싱
        06. 정렬
        07. 행렬 연산 
        08. Broadcasting
"""

def numpy_01():
    """
        Content
            01. 기본 사용법
        Describe

        Sub Contents
            1. 차원별 생성 방법
                1D array    :   x축 존재
                2D array    :   x, y축 존재
                3D array    :   x, y, z축 존재
            2. shape 다루기
                행렬의 차원을 shape라는 개념으로 다룸
                ndim    : 차원을 나타낸다.
            shape   : 행렬의 형태를 보여준다. (1,2)
    """
    import numpy as np
    print("\n", "=" * 5, "01. numpy의 기본구조 ", "=" * 5)
    print("\n", "=" * 3, "01_01. 차원별 생성 방법", "=" * 3)
    arr1D = np.array([1,2,3,4], dtype=int)
    arr2D = np.array([[1,2,3,4], [5,6,7,8]])
    arr3D = np.array([
            [[1, 2, 3, 4],      [5, 6, 7, 8],       [9, 10, 11, 12]],
            [[11, 12, 13, 14],  [15, 16, 17, 18],   [19, 20, 21, 22]]
        ]) 
        
    print('arr1D :', arr1D)
    print('arr2D :', arr2D)
    print('arr3D :', arr3D)

    print("\n", "=" * 3, "01_02. shape다루기", "=" * 3)
    print(arr1D.ndim , '차원', 'arr1D.shape :', arr1D.shape)
    print(arr2D.ndim , '차원', 'arr2D.shape :', arr2D.shape)
    print(arr3D.ndim , '차원', 'arr3D.shape :', arr3D.shape)

numpy_01()

def numpy_02():
    """
        Content
            02. DataType, 연산
        Describe
            numpy shape 에서는 하나의 단일 데이터 타입만 허용된다.

        Sub Contents
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
    import numpy as np
    print("\n", "=" * 5, "02. DataType, 연산", "=" * 5)
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

def numpy_03():
    """
        Content
            03. 인덱싱 슬라이싱
        Describe
            차원별로 인덱싱을 지정한다.

        Sub Contents

            01. 1차원
                기본 리스트 인덱싱과 동일하다.
                리스트 글 참조

            02. 2차원
                lst(x, y) 형태로 구성된다.
                x = 행, y는 열이 된다.
    """
    import numpy as np
    print("\n", "=" * 5, "03. 인덱싱 슬라이싱", "=" * 5)
    print("\n", "=" * 3, "03_01. 1차원 ", "=" * 3)

    lst1D = np.array([1,2,3,4])
    print('lst1D :', lst1D[:-1])

    print("\n", "=" * 3, "03_02. 2차원 ", "=" * 3)

    lst2D = np.array( [[1,2,3,4] , [5,6,7,8], [9,10,11,12]])
    print('lst2D[0, :] :', lst2D[0, :])
    print('lst2D[:, 0] :', lst2D[:, 0])
    print('lst2D[0:, :2] :', lst2D[0:, :2])
    print('lst2D[0:, :2] :', lst2D[2:, :])


def numpy_04():
    """
        Content
            04. Fancy 인덱싱
        Describe
            특정 index의 집합을 추출하고 싶을때 사용한다.

        Sub Contents
            01. 1차원

            02. 2차원
                lst(x, y) 형태로 구성된다.
                x = 행, y는 열이 된다.
    """
    import numpy as np
    print("\n", "=" * 5, "04. Fancy 인덱싱", "=" * 5)
    print("\n", "=" * 3, "04_01. 1차원 ", "=" * 3)

    idx = [1, 3, 5]
    lst1D = np.array([1,22,333,444,5555,66666,77777])
    print('lst1D[idx]', lst1D[idx])


    print("\n", "=" * 3, "04_02. 2차원 ", "=" * 3)

    lst2D = np.array( [[1,2,3,4] , [5,6,7,8], [9,10,11,12]])
    print('lst2D[[0,1], :] :', lst2D[[0,1], :])


def numpy_05():
    """
        Content
            05. Boolean 인덱싱
        Describe
            조건 필터링을 통하여 True, False로 색인한다.

        Sub Contents
            01. 기본 사용법
            02. 조건절 사용법
                lst[조건필터]
    """
    import numpy as np
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


def numpy_06():
    """
        Content
            06. 정렬
        Describe
            특정 index의 집합을 추출하고 싶을때 사용한다.
        Sub Contents
            01. 1차원 정렬
            02. n차원 정렬
    """
    import numpy as np
    print("\n", "=" * 5, "06. Boolean 인덱싱", "=" * 5)
    print("\n", "=" * 3, "06_01. 1차원 정렬 ", "=" * 3)
    lst = [1,5,6,8,9,4,2,7,6,3,8,33,40]

    np_lst = np.array(lst)
    print('오름차순 : ', np.sort(np_lst))
    print('내림차수 : ', np.sort(np_lst)[::-1])
    print('값이 유지되지는 않는다 : ', np_lst)
    np_lst.sort()
    print('np_lst.sort() : ', np_lst)

    print("\n", "=" * 3, "06_02. n차원 정렬 ", "=" * 3)

    np2d_lst = np.array([[5,6,7,8], [4,3,2,1], [10,12,11,9]])
    print('3,4 2차원 행렬 : ', np2d_lst.shape)
    print('행정렬 (0축정렬) : ', np.sort(np2d_lst, axis=0))
    print('열정렬 (1축정렬) : ', np.sort(np2d_lst, axis=1))


def numpy_07():
    """
        Content
            07. 행렬 (Matrix) 연산
        Describe
            특정 index의 집합을 추출하고 싶을때 사용한다.
        Sub Contents
            01. 덧셈, 곱샘 (shape을 맞춰야한다)
            02. sum
            03. np.dot : dot product 스칼라곱 = (3, 2) * (2, 3) = (3, 3)
    """
    import numpy as np
    print("\n", "=" * 5, "07. 행렬 (Matrix) 연산", "=" * 5)

    print("\n", "=" * 3, "07_01. 덧셈 ", "=" * 3)
    lst2d_01 = np.array([[1,2,3], [4,5,6]])
    lst2d_02 = np.array([[7,8,9], [10,11,12]])
    print('더하기 : ', lst2d_01 + lst2d_02)
    print('곱하기 : ', lst2d_01 * lst2d_02)

    print("\n", "=" * 3, "07_02. sum ", "=" * 3)
    print('행렬 내부 합산 :', np.sum(lst2d_01, axis=0))
    print('행렬 내부 합산 :', np.sum(lst2d_01, axis=1))

    print("\n", "=" * 3, "07_03. np.dot : dot product 스칼라곱 ", "=" * 3)
    lst2d_03 = np.array([[1,2], [3,4], [5,6]])
    lst2d_04 = np.array([[7,8,9], [10,11,12]])
    print('닿는 부분이 같아야한다. 결과는 바깥', lst2d_03.shape, lst2d_04.shape)
    print('곱하기 : ', np.dot(lst2d_03, lst2d_04))


def numpy_08():
    """
        Content
            08. Broadcasting
        Describe
            전체에 특정 값을 연산하여 사용할 수있다.
        Sub Contents
            01. 연산
    """
    import numpy as np
    print("\n", "=" * 5, "08. Broadcasting", "=" * 5)
    print("\n", "=" * 3, "08_01. 연산 ", "=" * 3)
    lst_2D = np.array([[1,2,3], [4,5,6]])
    print('lst_2D + 2 :', lst_2D + 2)
    print('lst_2D - 2 :', lst_2D - 2)
    print('lst_2D * 2 :', lst_2D * 2)
    print('lst_2D / 2 :', lst_2D / 2)
    print('lst_2D % 2 :', lst_2D % 2)