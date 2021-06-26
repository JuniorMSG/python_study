"""
    subject
        Data analysis module
        수학 과학 계산용 패키지
    topic
        numpy 모듈

    Contents
        01. 기본 사용법
        02. DataType, 연산
        ★ 03. 인덱싱 , 슬라이싱
        04. Fancy 인덱싱
        05. Bollean 인덱싱
        06. 정렬
        07. 행렬 연산
        08. Broadcasting
"""


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

numpy_03()