""" NW
    Subject
        built-in Function (내장 함수)

    Topic
        print() : 출력함수

    Describe
        print(DATA) 함수는 입력한 자료형의 데이터를 출력하는 것이다.
        DATA로 입력 받은 값을 출력한다.

    Contents
        01. 사용법
        02. 변수 출력방법
"""

# 01. 사용법
def print_01():
    """
    01. 사용법
        01_01. 구분자 사용법
            print("Hello World")
            print('Hello World')
            print('Hello', 'World')
            print('Hello' 'World')
        01_02. option
            end
                end옵션을 사용하면 그 뒤의 출력값과 이어서 출력한다. (줄바꿈 X)
            sep
                구분자를 지정해서 출력한다. default = ' '
                ,를 사용하면 자동으로 띄어쓰기가 되는데 이걸 바꿔주는 옵션
    """
    print("\n", "=" * 5, "01. 사용법", "=" * 5)

    print("\n", "=" * 3, "01_01. 구분자 사용법", "=" * 3)
    print("Hello World")
    print('Hello World')
    print('Hello', 'World')
    print('Hello' 'World')

    print("\n", "=" * 3, "01_02. option", "=" * 3)
    print('01_02. option=end')
    for cnt in range(1, 10):
        print(cnt, end=' ')
    print('')
    print('01_02.', 'option=sep' , '999', sep=' sepTest ')

# print_01()


#02. 변수 출력방법
def print_02():
    """
        02. 변수 출력방법 - Formatting
            02_01. % 출력 방식
            02_02. .format
            02_03. f - format
    """

    print("\n", "=" * 5, "02. 변수 출력방법 - Formatting", "=" * 5)

    print("\n", "=" * 3, "02_01. % 출력 방식", "=" * 3)
    print("%d %f" % (1, 2.2))
    print("%d%% %f%%" % (1, 2.2))
    last_name, first_name = '김', '철수'
    print("나는 %s씨 가문의 %s 입니다" % (last_name, first_name))

    print("\n", "=" * 3, "02_02. .format", "=" * 3)
    print("{0} {1}".format(8, 5.2))
    print("{number} {float}".format(float=5.2, number=8))
    print("나는 {0}씨 가문의 {1} 입니다".format(last_name, first_name))
    print("나는 {last_name}씨 가문의 {first_name} 입니다".format(first_name=first_name, last_name=last_name))

    print("\n", "=" * 3, "02_03. f - format", "=" * 3)

    dict_name = {"last_name": last_name, "first_name": first_name}
    print('f문자열 포매팅 :', f'나는 {last_name}씨 가문의 {first_name} 입니다')
    print('f문자열 포매팅 :', f'나는 {dict_name["last_name"]}씨 가문의 {dict_name["first_name"]} 입니다')

    print('f문자열 포매팅 :', f'나는 {dict_name["last_name"]:<10}씨')
    print('f문자열 포매팅 :', f'나는 {dict_name["last_name"]:^10}씨')
    print('f문자열 포매팅 :', f'나는 {dict_name["last_name"]:>10}씨')
    print('f문자열 포매팅 :', f'나는 {dict_name["last_name"]:=>10}씨')
    print('f문자열 포매팅 :', f'나는 {dict_name["last_name"]:!>10}씨')

print_02()