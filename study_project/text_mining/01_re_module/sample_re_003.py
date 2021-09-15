# 파이썬 re 모듈의 다양한 함수 - match, findall, search
# + : 기호 앞에 표시된 문자가 연속적으로 1이상이 있는지를 검색 --> 1이상이면 모두 Ok
# 그래서 앞서 전화번호 첫자리가 45678541 이렇게 길어도 검색을 한 것임.
import re

# findall : 대상 문자열(텍스트)에서 패턴 조건에 맞는 문자열을 찾아서 그 값들을 리스트로 반환.
# findall 함수는 검색 패턴이 겹치지 않는 매칭을 한다.
rst_matchObj1 = re.findall( 'k2k', 'k2k2k2k2' )
print( "findall 검색 결과 : ", rst_matchObj1 )

# search 함수는 match 함수처럼 처음부터 매칭을 검사하지만 조건에 맞는 패턴이 발견되면 더이상 하지 않는다.
# 따라서 처음에 매칭되는 패턴이 나와버리면 match 함수와 결과가 같다.
rst_matchObj2 = re.search( 'k2k', 'k2k2k2k2' )
rst_matchObj3 = re.match( 'k2k', 'k2k2k2k2' )
print( "search 검색 결과 : ", rst_matchObj2 )
print( "match 검색 결과 : ", rst_matchObj3 )
print( "------------------------------------------------" )

rst_matchObj4 = re.search( 'k2k', '2k2k2k2k' )  # 1, 4     k2k
rst_matchObj5 = re.match( 'k2k', '2k2k2k2k' )  # None
print( "search 검색 결과 : ", rst_matchObj4 )
print( "match 검색 결과 : ", rst_matchObj5 )



# Summary
# findall vs match, search 함수는 반환하는 결과 값이 서로 다르다.
# match, search 함수는 matchObject 객체로 결과를 반환.
# findall --> 리스트로 반환.
# 그래서 콘솔창에 출력되는게 서로 틀리다.







