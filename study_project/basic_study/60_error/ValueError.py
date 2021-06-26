
"""
    Subject
        ERROR 정리
    Topic
        ValueError
    Content
        ValueError: invalid literal for int() with base 10: '3.14'
    Describe
        파이썬에서 데이터 형 변환시 발생하는 에러
"""

dtype_int = 1
dtype_float = 3.14
dtype_str = '3.14'

print(int(dtype_int),   float(dtype_int),   str(dtype_int))
print(int(dtype_float), float(dtype_float), str(dtype_float))
print(float(dtype_str),   str(dtype_str))
print(int(float(dtype_str)))