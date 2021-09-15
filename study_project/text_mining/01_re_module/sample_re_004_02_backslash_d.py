# 문자열내에서 /d 자체를 검색하고자 할 때와 숫자를 검색할 때 주의점 비교
import re

# (Question) 다음중 에러가 나는 것은?
# '\s'     '\'     's\'
# SyntaxError : ............................................


# [1]
# 메타문자 --->       \     $       .       ^       +       *       ?       {       }       [       ]       (       )       |
rst_matchObj1 = re.search( 'a\\d', 'a\\d' )  # None
print( "[1] : ", rst_matchObj1 )
#       a\n    -->    정규식 엔진 내부에서 search() 함수가 백슬래시(\) 해석을 일반 문자가 아닌 메타문자(특별한)로 해석함.
#       a\n    -->    파이썬 인터프리터에 의해서 백슬래시(\)를 일반문자로 해석함.

# Summary
# 패턴을 일치시키는 방법은?
# (1) 'a\\\\d'
# (2) r'a\\d'


# [2]
text2 = "정규식에서 \d는 숫자를 의미한다. 숫자란 123, 456, 789 이런걸 말한다."
pattern2 = re.compile( r'\d' )
rst_matchLst2 = re.findall( pattern2, text2 )
print( "[2] : ", rst_matchLst2 )


# [3] : 숫자가 아닌 \d 자체를 검색한다면? ( r선언을 하지 않는다면? )
text3 = "정규식에서 \d는 숫자를 의미한다. 숫자란 123, 456, 789 이런걸 말한다."
pattern3 = re.compile( r'\\d' )
rst_matchObj3 = re.search( pattern3, text3 )
print( "[3] : ", rst_matchObj3 )















