"""
    subject
        Data analysis module
        수학 과학 계산용 패키지
    topic
        numpy 모듈

    Contents
        01. 기본 사용법
        02. DataType, 연산
        03. 인덱싱 , 슬라이싱
        ★ 04. Fancy 인덱싱
        05. Bollean 인덱싱
        06. 정렬
        07. 행렬 연산
        08. Broadcasting
"""

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

numpy_04()