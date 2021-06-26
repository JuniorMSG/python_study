"""
Author  : SG
DATE    : 2021-05-18
02_Data_Type

★ 02_01. Integer   (숫자형)
02_02. String       (문자열)
02_03. Boolean      (참&거짓 자료형)
02_04. List         (리스트)
02_05. Tuple        (수정 불가능 리스트)
02_06. Dictionary   (HashMap - Key- Map Type)
02_07. Set          (집합 자료형)
02_08. Variable     (변수)

★ 02_01. Integer   (숫자형)
    01. Integer     숫자형
    02. Operator    연산자
"""


"""
01. Integer
숫자형이란?
숫자 형태로 이루어진 자료형 - 정수, 실수, 2진수, 8진수 등등을 뜻함
"""
print("=" * 5, "01. Integer", "=" * 5)

print('정수', 123, -345, 0)
print('실수', 1.234, -3.44)
print('2진수', bin(4), bin(9))
print('8진수', oct(4), oct(9))
print('16진수', hex(4), hex(9))


"""
02. Operator
연산자란?
기본적인 사칙연산 (+ - * /) 부터 제곱 (**), 나머지 반환 (%), 몫 반환 (//) 등을 말함 
"""
print("=" * 5, "02. Operator", "=" * 5)
num1, num2 = 3, 7
print("Plus     :   ", num1 + num2)
print("Minus    :   ", num1 - num2)
print("Multiple :   ", num1 * num2)
print("Division :   ", num1 / num2)
print("제곱     :   ", num1 ** num2)
print("나머지   :   ", num1 % num2)
print("몫       :   ", num1 // num2)