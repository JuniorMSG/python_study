"""
    02_data_structure
        ▶ 01. list
        02. tuple
        03. Dictionary
        04. Set
"""

"""
      01. list
            01_01. 리스트 기본
                  [] 대괄호 사용
            01_02. 리스트 컨트롤
            01_03. 리스트 메소드
            01_04. 리스트 복사
            01_05. 리스트 예시
            
            
            
      help(list) DOC 호출
"""

print('='*20, '01. list', '='*20)

print('='*5, '01_01. 리스트 기본', '='*5)
lst = [1, 5, 10 ,15 ,20 ,25 ,30 ,35, 40]
print(type(lst))
print(lst)
print(lst[0:3])
print(lst[::3])
print(lst[::-1])

lst_str = ['a', 'b', 'c']

x = [lst, lst_str]
print(x)
print(x[0][1])
print(x[0][::3])
print(x[1][::2])


print('='*5, '01_02. 리스트 컨트롤', '='*5)

s = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
print(s[0])
s[0] = 'X'
print(s)
s.append(100)
print('s.append(100)' , s)
print('s.pop()', s.pop(), '=> : ', s)
print('s.pop(0)', s.pop(0), '=> : ', s)

del s[0]
print('del s[0]', '=> : ', s)

# s list 자체가 삭제됨
# del s
# print('del s', '=> : ', s)

s.remove('d')
print('s.remove(d)',  '=> : ', s)
s.append('d')
s.append('d')
s.append('d')

s.remove('d')
print('s.remove(d)',  '=> : ', s)

lst_01 = [1, 2, 3, 4, 5]
lst_02 = [6, 7, 8, 9, 10]
lst_01.extend(lst_02)
print('lst_01.extend(lst_02) =>', lst_01)


print('='*5, '01_03. 리스트 메소드', '='*5)
r = [1, 2, 3, 4, 5, 6, 7, 8, 3]
print(r.index(3, 3))
print(r.count(3))

if 5 in r :
      print('exist')

r.sort()
print(r)
r.sort(reverse=True)
print(r)
r.reverse()
print(r)

s = 'my name is sg'
to_split = s.split(' ')
print(to_split)

x = ' %%% '.join(to_split)
print(x)

print(help(list))


print('='*5, '01_04. 리스트 복사', '='*5)

# 참조 전달
i = [1, 2, 3, 4, 5]
j = i
j[0] = 100
print('j =', j)
print('i =', i)

# 수치 전달
k = i.copy()
k[0] = 50
print('i =', i)
print('k =', k)

# 수치 전달
X = 20
Y = X
Y = 5
print(X)
print(Y)
print('X =', id(X))
print('Y =', id(Y))

# 참조 전달
X = [ 'a', 'b' ]
Y = X
Y[0] = 'p'
print(X)
print(Y)
print('X =', id(X))
print('Y =', id(Y))


"""
      01_05. 리스트 예시
      Taxi Driver
"""

print('='*5, '01_05. 리스트 예시', '='*5)
seat = []
min = 0
max = 5

seat.append(1)
print(min <= len(seat) < max)
seat.append(1)
print(min <= len(seat) < max)
seat.append(1)
print(min <= len(seat) < max)
seat.append(1)
print(min <= len(seat) < max)
seat.append(1)
print(min <= len(seat) < max)




