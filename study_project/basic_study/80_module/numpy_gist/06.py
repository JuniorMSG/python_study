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
        ★ 06. 정렬
        07. 행렬 연산
        08. Broadcasting
"""


def numpy_06():
    """
        Content
            06. 정렬
        Describe
            특정 index의 집합을 추출하고 싶을때 사용한다.
        Sub Contents
            01. 1차원 정렬
                - 인덱싱 방법
                - sort()
            02. n차원 정렬
    """
    import numpy as np
    print("\n", "=" * 5, "06. Boolean 인덱싱", "=" * 5)
    print("\n", "=" * 3, "06_01. 1차원 정렬 ", "=" * 3)
    lst = [1,5,6,8,9,4,2,7,6,3,8,33,40]

    np_lst = np.array(lst)
    print('오름차순 : ', np.sort(np_lst))
    print('내림차수 : ', np.sort(np_lst)[::-1])
    print('인덱싱 방법으로는 값이 유지되지는 않는다 : ', np_lst)

    np_lst.sort()
    print('np_lst.sort() : ', np_lst)

    print("\n", "=" * 3, "06_02. n차원 정렬 ", "=" * 3)

    np2d_lst = np.array([[5,6,7,8], [4,3,2,1], [10,12,11,9]])
    print('3,4 2차원 행렬 : ', np2d_lst.shape)
    print('행정렬 (0축정렬) : ', np.sort(np2d_lst, axis=0))
    print('열정렬 (1축정렬) : ', np.sort(np2d_lst, axis=1))

numpy_06()