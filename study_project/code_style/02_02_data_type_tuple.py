"""
    02_data_structure
        01. list
        ▶ 02. tuple
        03. Dictionary
        04. Set
"""

"""
      02. tuple
            변경하지 않으려는 값은 튜플로 사용해야 좋다.
            02_01. 튜플 기본
                  () 괄호 사용
            02_02. 튜플 언패킹
            02_03. 사용 예시
            
      print(help(tuple)) DOC 호출
"""

print('='*20, '02. tuple', '='*20)
print('='*5, '02_01. 튜플 기본', '='*5)
t = (1, 2, 3, 4, 5, 6)

# t[0] = 100 Error
print(t[0])
print(t[0:2])
print(t[::2])
print(t[::-1])


print('='*5, '02_02. 튜플 언패킹', '='*5)

num_tuple = (10, 20)
print(num_tuple)

x, y = num_tuple
print(x, y)

x, y = 10, 20
print(x, y)

min, max = 0, 100
print(min, max)

# 변수 만들기.
a, b, c, d, e, f = 1, 2, 3, 4, 5, 6
a = 'KK'
b = 999
print(a, b, c, d, e, f)

# 변수 변경하기 원래라면
i = 10
j = 20
print(i, j)
tmp = i
i = j
j = tmp
print(i, j)
# 파이썬
a = 100
b = 200
print(a, b)
a, b = b, a
print(a, b)


print('='*5, '02_03. 사용 예시', '='*5)
chose_from_two = ('A', 'B', 'C')

answer = []
answer.append('A')
answer.append('B')

print(chose_from_two)
print(answer)

