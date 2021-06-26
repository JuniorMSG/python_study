"""
    subject
        Data analysis module
        수학 과학 계산용 패키지
    topic
        numpy 모듈

    Contents
        01. 기본 사용법
        ★ 02. DataType, 연산
        03. 인덱싱 , 슬라이싱
        04. Fancy 인덱싱
        05. Bollean 인덱싱
        06. 정렬
        07. 행렬 연산 
        08. Broadcasting
"""


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
    arr3 = np.array([1, 2, '3.14', '문자열', 4, 5.5])
    print(arr3)
    print("문자열 취급받아서 연산된다. : ", arr3[0] + arr3[1])

    print("\n", "=" * 3, "02_04. int + str    dtype 지정시", "=" * 3)
    print("변환이 불가능한 경우 변환되지 않는다.")
    # invalid literal for int() with base 10: '3.14' 에러 발생
    # print(np.array([1, 2, '3.14', '문자열',4, 5.5], dtype=int))