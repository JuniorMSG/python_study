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
        ★ 05. Bollean 인덱싱
        06. 정렬
        07. 행렬 연산
        08. Broadcasting
"""


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
    print(lst1D>2)
    print('조건절 인덱싱 lst1D[lst1D>2] : ', lst1D[lst1D>2])

numpy_05()