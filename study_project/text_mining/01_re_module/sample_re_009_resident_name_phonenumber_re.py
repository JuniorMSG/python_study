# 이름과 전화번호가 들어있는 텍스트에서 전화번호가 맞는지 검증하여 출력하시오?
text = '''홍길동 010-1234-5671
이순신 010-3333-9632
강감찬 010_2222_4122
김유신 010.9999.5417
을지문덕 010 119 2222'''

# [1] : re 모듈 삽입
import re


# [2] : 조건에 맞는 패턴 컴파일
#pattern = re.compile( r'^\w+\s+\d{3}-\d{3,4}-\d{4}$', re.M )
#rst_matchLst = re.findall( pattern, text )
#print( rst_matchLst )


# [3] : 전화번호 구분자 -->     '-'     ' '     '.'     '_'
#pattern = re.compile( r'^\w+\s+\d{3}[-\s._]+\d{3,4}[-\s._]+\d{4}$', re.M )
#rst_matchLst = re.findall( pattern, text )
#print( '\n'.join(rst_matchLst) )


# [4] : 그루핑
# 그루핑 --> (1)먼저 매치가 된 결과물을 가지고 (2)그루핑한 것에 대해서 재매치.
#pattern = re.compile( r'^(\w+)\s+(\d{3})[-\s._]+(\d{3,4})[-\s._]+(\d{4})$', re.M )
#rst_matchLst = re.findall( pattern, text )
#print( rst_matchLst )


# [5] : 그루핑된 문자열을 sub 메서드 사용해서 다른 문자열로 바꾸기
# sub 메서드를 사용하면 매치된 부분과 순서를 다르게 출력하거나 또는 다른 문자로 대체해서 출력시킬 수도 있다.
#pattern = re.compile( r'^(\w+)\s+(\d{3})[-\s._]+(\d{3,4})[-\s._]+(\d{4})$', re.M )
#print( pattern.sub( r'\g<1> : \g<2> - \g<3> - ****', text ) )
#print( re.sub( r'(?m)^(\w+)', '***', text ) )


# [6] : re.sub() 메서드내에서 치환 카운트 사용
#print( re.sub( pattern=r'aaa', repl='bbb', count=3, string='aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa' ) )
print( re.sub( r'aaa', 'xxx', 'aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa', 9 ) )
















