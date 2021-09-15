# 파이썬 re 모듈의 match() 인자
# 3번째인자 : 대소문자 무시 -> re.I (대문자 I)

# 검색 패턴 매치
# [1]
rst_matchObj1 = re.match( 'kor', 'Korea', re.I )
print( "[1] : ", rst_matchObj1 )
print( "----------------------------------------------------" )

# [2] : 인라인 방식
#rst_matchObj2 = re.match( (?i)'kor', 'Korea' )  #Error
rst_matchObj2 = re.match( '(?i)kor', 'Korea' )
print( "[2] : ", rst_matchObj2 )
print( "----------------------------------------------------" )

# [3]
# ignore (v)무시[묵살]하다 (=disregard 무시, 묵살(n), 무시[묵살]하다(v)
#rst_matchObj3 = re.match( 'kor', 'Korea', re.ignorecase )  #Error
rst_matchObj3 = re.match( 'kor', 'Korea', re.IGNORECASE )
print( "[3] : ", rst_matchObj3 )
print( "----------------------------------------------------" )







