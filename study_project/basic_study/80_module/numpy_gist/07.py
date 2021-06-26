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
        04. Fancy 인덱싱
        05. Bollean 인덱싱
        06. 정렬
        ★ 07. 행렬 (Matrix) 연산
        08. Broadcasting
"""


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

numpy_07()