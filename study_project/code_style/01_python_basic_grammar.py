"""
    Python 기본
        01. 변수 선언
        02. 연산
        03. 문자열
"""

"""
      01. 변수 선언 
"""
num: int = 1
name = 'SG'

print(type(num))
print(type(name))
print('Hi', 'SG', sep=',')


"""
      02. 연산
      help(math)  : 헬프 함수 사용시 해당 클래스의 도움말을 볼 수 있음.
"""
print(5+5)
print(5*5)
print(5/5)
pie = 3.141592
print(round(pie, 2))
print(type(num))

import math
print(math.sqrt(25))

y = math.log2(10)
print(y)



"""
      03. 문자열
            03_01. 기본출력
            03_02. 인덱싱, 슬라이싱 
            03_03. Method
            03_04. 문자열 대입
"""

print('='*20, '03. 문자열', '='*20)
print('='*5, '03_01. 기본출력', '='*5)
print('hello')
print("hello")
print("""
hello
SG
MS
""")

s = ( 'aaaaaaaaaaaaa' 'bbbbbbbbbbbbbb'
      'ccccccccccccc' 'dddddddddddddd')

print('='*5, '03_02. 인덱싱, 슬라이싱 ', '='*5)

text = 'hello world'
print(text[1])
print(text[-1])
print(text[0:5])
print(text[5:9])

print('='*5, '03_03. Method ', '='*5)
print("text.startswith('hello') : ", text.startswith('hello'))
print("text.find('world') : ", text.find('world'))
print("text.count('world') : " , text.count('world'))
print("text.capitalize() : ", text.capitalize())
print("text.title() : ", text.title())
print("text.upper() : ", text.upper())
print("text.lower() : " , text.lower())
print("text.replace('hello', 'Hello Python') : ", text.replace('hello', 'Hello Python'))
print(text)

print('='*5, '03_04. 문자열 대입 ', '='*5)

print('='*3, 'Python 3.6 이전 format 방식', '='*3)
print('a is {}'.format('a'))
print('a is {} {} {}'.format(1, 2, 3))
print('a is {0} {1} {2}'.format('my', 'name', 'is'))
print('my name {a} {b} {c}'.format(a='is', b='sg', c='ms'))

print('='*3, 'Python 3.6 이후 사용 가능 방식 f-string', '='*3)
a = 'a'
print(f'a is {a}')

x, y, z = 1, 2, 3
print(f'a is {x}, {y}, {z}')
print(f'a is {z}, {y}, {x}')

name = 'sg'
family = 'm'
print(f'My name is {name} {family}. I am {family} {name}')


