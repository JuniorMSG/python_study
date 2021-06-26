"""
    arange 
        순서대로 리스트에 값을 생성하는 numpy 함수

    range
        파이썬 내장함수

    01. 사용법
        01. 기본 사용법
        02. 차이점
    02. 예시 
"""

"""
    01. 사용법
        1. 기본 사용법
            arange(start이상, stop미만, (증가값))   or arange(start=1, stop=11, (증가값))
            range(start이상, stop미만, (증가값))   or arange(start=1, stop=11, (증가값))
            기본 증가값은 1입니다.

        2. 차이점
            python3가 되면서 range함수의 리턴값은 range이기 때문에 for문에서 범위를 지정하는 구문으로 활용된다 
            for문에 활용할 경우 같지만 
            리스트형태로 만들어 내려면 range함수는 부적절하다.    
"""
import numpy as np
print("\n", "=" * 5, "01_01. 기본 사용법 ", "=" * 5)

for data in range(1, 11):
    print(data)

lst_numpy_arange = np.arange(1, 11)
print('lst_numpy_arange :', lst_numpy_arange)

print("\n", "=" * 3, "01_02. 차이점 ", "=" * 3)
print('lst_range :', type(range(1, 11)))
print('lst_numpy_arange :', type(lst_numpy_arange))

"""
    02. 예시
        range는 for문에서, arange는 리스트형태로 만들때 사용합니다. 
        arange도 실제 데이터 타입은 numpy.ndarray 입니다.
        
        1. 홀수 값만 생성
"""

print("\n", "=" * 5, "02. 예시 ", "=" * 5)
print("\n", "=" * 3, "02_01. 홀수 값만 생성 ", "=" * 3)

for data in range(1, 11, 2):
    print(data)

lst_numpy_arange = np.arange(1, 11, step=2)
print(lst_numpy_arange)

