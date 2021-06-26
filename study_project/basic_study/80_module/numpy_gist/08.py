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
        07. 행렬 연산
        ★ 08. Broadcasting
"""


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



numpy_08()