"""
03. 연산자
    03-01. 비교, 관계 연산자
    03-02. 논리 연산자
    03-03. 포함 연산
"""

print("\n", "=" * 5, "03. 연산자", "=" * 5)

print("\n", "=" * 3, "03_01. 비교, 논리 연산자", "=" * 3)
print("3 > 5  : x가 y보다 크다", 3 > 5)
print("3 < 5  : x가 y보다 작다", 3 < 5)
print("3 >= 5 : x가 y보다 크거나 같다.", 3 >= 5)
print("3 <= 5 : x가 y보다 작거나 같다.", 3 <= 5)
print("3 == 5 : x와 y가 같다.", 3 == 5)
print("3 != 5 : x와 y가 같지 않다.", 3 != 5)

print("\n", "=" * 3, "03_02. and or not", "=" * 3)
print("x or  y : 둘중 하나만 참이면 참:", True or False)
print("x and y : 둘다 참이여야 참:", True and True)
print("not x : x가 거짓이면 참이다.:", not False)


print("\n", "=" * 3, "03_03. in, not in", "=" * 3)
data = [1, 2, 3]
print("3 in data : 3이 데이터 안에 포함된다. ", 3 in data)
print("3 not in data : 3이 데이터 안에 포함되지 않는다. ", 3 not in data)