# 연습문제
# 아래의 정규식 패턴에 문제가 있는지 살펴보시오? 에러가 난다면 왜 에러가 나는지 말해보시오?
import re

text = 'my.name@localhost.com'

pattern = re.compile( r'[a-zA-Z0-9-.]+@[\w-.]+.com' )
rst_matchLst = re.findall( pattern, text )
print( rst_matchLst )


# Summary
# 이런 문제는 주로 [\w-_] 문자 클래스 내부에 하이픈(-)이 있기 때문에 발생하는 경우가 많다.
# 하이픈(-)은 문자 간격으로 해석될 수 있으므로 하이픈(-)을 문자 자체로 사용하려면 모호함을 방지하기 위해서
# 첫번째 또는 마지막 위치에 하이픈(-)을 넣거나 이스케이프 처리하는 것이 좋다.



