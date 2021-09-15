# 파이썬 re 모듈의 함수 인자
# 3번째인자 : 마침표 -> 개행문자(\n)와 일치 여부 -> re.S
import re

# 검색 패턴 매치
# [1]
rst_matchObj = re.findall( 'k.', 'kr k k\n ks' )                 # kr, k_, ks
rst_matchObj1 = re.findall( 'k.', 'kr k k\n ks', re.S )      # kr, k_, k\n, ks

# [2]
rst_matchObj2 = re.findall( '(?s)k.', 'kr k k\n ks' )

# [3]
rst_matchObj3 = re.findall( 'k.', 'kr k k\n ks', re.DOTALL )


# 출력
print( "[0] : ", rst_matchObj )
print( "--------------------------------------------" )
print( "[1] : ", rst_matchObj1 )
print( "--------------------------------------------" )
print( "[2] : ", rst_matchObj2 )
print( "--------------------------------------------" )
print( "[3] : ", rst_matchObj3 )



# Summary
# 대상이 되는 문자열(텍스트)내의 개행문자를 마침표에 일치시킬 것인지 아닌지를 3번째 인자의 re.S 옵션 설정을 통해서 할 수 있다.
# 문자열(텍스트)을 행 단위로 검색하면서 패턴을 검색한다든지 할 때 사용할 수 있다.
# 또는 전체 문자열(텍스트)을 대상으로 할 때도 사용할 수 있다.


