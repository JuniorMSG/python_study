"""
    subject
        Data analysis module
        수학 과학 계산용 패키지
    topic
        numpy 모듈

    Contents
        ★ 01. numpy의 기본구조
        02. DataType, 연산
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
    arr1D = np.array([1, 2, 3, 4], dtype=int)
    arr2D = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    arr3D = np.array([
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]]
    ])

    print('arr1D :', arr1D)
    print('arr2D :', arr2D)
    print('arr3D :', arr3D)

    print("\n", "=" * 3, "01_02. shape다루기", "=" * 3)
    print(arr1D.ndim, '차원', 'arr1D.shape :', arr1D.shape)
    print(arr2D.ndim, '차원', 'arr2D.shape :', arr2D.shape)
    print(arr3D.ndim, '차원', 'arr3D.shape :', arr3D.shape)


numpy_01()