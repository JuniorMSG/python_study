"""
Author  : SG
DATE    : 2021-05-18
02_Data_Type

02_01. Integer      (숫자형)
★ 02_02. String    (문자열)
02_03. Boolean      (참&거짓 자료형)
02_04. List         (리스트)
02_05. Tuple        (수정 불가능 리스트)
02_06. Dictionary   (HashMap - Key- Map Type)
02_07. Set          (집합 자료형)
02_08. Variable     (변수)

★ 02_02. String   (문자열)
    01. String      문자열이란?
    02. How To Use     
    03. 문자열에 ', " 포함하기
    04. 2줄 이상 문자열 사용
    05. 문자열 연결(Concatenation) 
    06. 인덱싱(Indexing)
    07. 슬라이싱(Slicing)
    08. 포매팅(Formatting)
    09. 내장함수(built-in Function)
    10. 이스케이프 시퀀스 (Escape Sequence)
"""

"""
01. String 
    01. 문자열이란? 문자, 단어 등으로 구성된 문자들의 집합을 의미한다.
"""


"""
02. How To Use
    01. 사용방법 : "", '', ''' ''', """ """에 둘러싸여 있으면 문자열이다.
"""
print()
print("=" * 5, "02. How To Use", "=" * 5)
print("문자열", '문자열', '''문자열''', """ 문자열 """)


"""
03. 문자열에 ', " 포함하기
    01. \' \" 역슬래시로 표현하기
    02. ""안에 ' 넣기
    03. ''안에 ' 넣기
    04. ''' '''안에 ', " 넣기
    05. ""안에 ', " 넣기
"""
print()
print("=" * 5, "03. 문자열에 \', \" 포함하기", "=" * 5)
print('01. \'\', \"\", \'\'\' \'\'\' \"\"\" \"\"\"')
print('02. " ')
print("03. ' ")
print('''04. ", ' ''')
print("""05. ", ' """)


"""
04. 2줄 이상 문자열 사용
    01. 더블 쿼테이션 3개 혹은 싱글 쿼테이션 3개 세트로 묶어서 표현하면 된다. (""" """, ''' ''')
"""
print()
print("=" * 5, "04. 2줄 이상 문자열 사용", "=" * 5)
String_Line = """===
    2줄 이상 문자열 사용하기 
=== """
print(String_Line)

"""
05. 문자열 연산 (Concatenation)
    01. Plus 
    02. * (Multiple)
"""
print("=" * 5, "05. 문자열 연산 (Concatenation)", "=" * 5)
String_Conc1 = 'Python'
String_Conc2 = 'Study'
String_Conc = String_Conc1 + String_Conc2
print('01. Plus : ', String_Conc)
print('02. * : ', String_Conc1 * 2, String_Conc2 *3)


"""
06. 인덱싱(Indexing) 
    찾아보기 쉽도록 일정한 순서로 나열한 목록을 뜻한다. 
    01. 사용방법1. 값[숫자] 형태로 사용 (양수는 앞에서부터 음수는 뒤에서부터)
"""
print("=" * 5, "06. 인덱싱(Indexing)", "=" * 5)
index_val = "My Life For Aiur"
print('01. 사용방법1', index_val[1], index_val[-1])


"""
07. 슬라이싱(Slicing)
    인덱싱에서 문자열을 뽑는 방법이다.
    01. index_val[0:5]  : 0에서 5까지
    02. index_val[0:]   : 0에서 끝까지
    03. index_val[:12]  : 처음부터 12번째 까지
    04. index_val[5:-3] : 5에서 -3까지
    05. 문자열 교체하기 
"""
print("=" * 5, "07. 슬라이싱(Slicing)", "=" * 5)
index_val = "My Life For Aiur"
print('01. index_val[0:5]', index_val[0:5])
print('02. index_val[0:]', index_val[0:])
print('03. index_val[:12]', index_val[:12])
print('04. index_val[5:-3]', index_val[5:-3])
print('05. 문자열 교체하기', index_val[:12] + "Earth")

"""
08. 포매팅(Formatting)
    문자열 사이에 값을 삽입하는 방법을 말한다.
    01. 1개 : % ,  복수개 : % ()
    02. .format
    03. 문자열 포맷 코드 활용 %
    04. 문자열 포맷 코드 활용 .format
    05. f문자열 포매팅 (파이썬 3.6이상)
"""
last_name = "김"
first_name = "철수"

print("=" * 5, "08. 포매팅(Formatting)", "=" * 5)
print("01.", "%d %f" % (1, 2.2))
print("01.", "%d%% %f%%" % (1, 2.2))
print("01. 나는 %s씨 가문의 %s 입니다" % (last_name, first_name))


print("02.", "{0} {1}".format(8, 5.2))
print("02.", "{number} {float}".format(float=5.2, number=8))
print("02. 나는 {0}씨 가문의 {1} 입니다".format(last_name, first_name))
print("02. 나는 {last_name}씨 가문의 {first_name} 입니다".format(first_name = first_name, last_name = last_name))


print("03. 오른쪽정렬   : %10s right" % last_name)
print("03. 왼쪽정렬     : %-10s left" % last_name)
print("03. 소수점       : %0.4f" % 3.141592)

print("04. 오른쪽정렬   : {0:>10} right".format(last_name))
print("04. 가운데정렬   : {0:^10} right".format(last_name))
print("04. 왼쪽정렬     : {0:<10} right".format(last_name))
print("04. 공백채우기   : {0:=^10} right".format(last_name))
print("04. 공백채우기   : {0:!>10} right".format(last_name))
print("04. 소수점       : {0:10.4f}".format(3.141592))
print("04. {{}} 사용하기: {{}}".format())

dict_name = {"last_name":last_name, "first_name":first_name}
print('05. f문자열 포매팅 :', f'나는 {last_name}씨 가문의 {first_name} 입니다')
print('05. f문자열 포매팅 :', f'나는 {dict_name["last_name"]}씨 가문의 {dict_name["first_name"]} 입니다')

print('05. f문자열 포매팅 :', f'나는 {dict_name["last_name"]:<10}씨')
print('05. f문자열 포매팅 :', f'나는 {dict_name["last_name"]:^10}씨')
print('05. f문자열 포매팅 :', f'나는 {dict_name["last_name"]:>10}씨')
print('05. f문자열 포매팅 :', f'나는 {dict_name["last_name"]:=>10}씨')
print('05. f문자열 포매팅 :', f'나는 {dict_name["last_name"]:!>10}씨')


"""
09. 내장함수(built-in Function)
    문자열 자료형 자체적으로 가지고 있는 함수
    01. 위치를 알려주는 함수    (find, index)
         find 없으면 -1, index 없으면 에러
    02. 문자열을 삽입 함수      (join)
    03. 대소문자 전환 함수      (upper, lower)
    04. 공백 제거 함수          (lstrip, rstrip, strip)
    05. 문자열 변경             (replace)
    06. 문자열 나누기           (split)
    07. 문자 개수 세기          (count)
"""


print("=" * 5, "09. 내장함수(built-in Function)", "=" * 5)
str = ' built In Function '
str2 = '-'

print("str.find('Q')   : ", str.find('Q'))
print("str.find('F')   : ", str.find('F'))
print("str.index('l')  : ", str.index('l'))
print("str.index('Q')  : error")

print("str.join(str2)  : ", str2.join(str))

print("str.upper()              : ", str.upper())
print("str.lower()              : ", str.lower())

print("str.lstrip() : ", str.lstrip())
print("str.rstrip() : ", str.rstrip())
print("str.strip()  : ", str.strip())

print("str.replace('In', 'Out') : ", str.replace('In', "Out"))
print("str.split()              : ", str.split())

print("str.count('i')  : ", str.count('i'))
print("str.count('Fn') : ", str.count('Fn'))




"""
10. 이스케이프 시퀀스 (Escape Sequence)
    프로그래밍할 때 사용할 수 있도록 미리 정의해둔 문자조합
    코드	설명
    \n	    문자열 안에서 줄을 바꿀 때 사용
    \t	    문자열 사이에 탭 간격을 줄 때 사용
    \\	    문자 \를 그대로 표현할 때 사용
    \'	    작은따옴표(')를 그대로 표현할 때 사용
    \"	    큰따옴표(")를 그대로 표현할 때 사용
    \r	    캐리지 리턴(줄 바꿈 문자, 현재 커서를 가장 앞으로 이동)
    \f	    폼 피드(줄 바꿈 문자, 현재 커서를 다음 줄로 이동)
    \a	    벨 소리(출력할 때 PC 스피커에서 '삑' 소리가 난다)
    \b	    백 스페이스
    \000	널 문자
"""

