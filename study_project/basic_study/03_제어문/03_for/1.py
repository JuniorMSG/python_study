"""
        01_01. 기본구조
            기본구조
            for 변수 in 리스트(튜플, 문자열등):
                수행할 문장~
        01_02. 리스트
        01_03. 문자열
        01_04. range()
"""
print("\n", "=" * 5, "01. 기본 사용방법", "=" * 5)
print("\n", "=" * 3, "01_02. 리스트", "=" * 3)
# 리스트 사용
lst_01 = [1, 2, 3]
for data in lst_01:
    print(data)

# 2중 리스트 사용
lst_02 = [[1,2], [3,4], [5,6]]
for data1, data2 in lst_02:
    print("data1 :", data1)
    print("data2 :", data2)

print("\n", "=" * 3, "01_03. 문자열", "=" * 3)
str_01 = "test"
# 문자열 사용
for data in str_01:
    print(data)

print("\n", "=" * 3, "01_04. range", "=" * 3)
# range 함수사용
for data in range(10):
    print(data)