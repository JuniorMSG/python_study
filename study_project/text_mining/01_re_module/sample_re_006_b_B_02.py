# 문자열내에서 특정 단어가 들어간 문자열 모두를 찾는 패턴은?
import re


# [1]
text1 = "(Question) kor 단어는 매칭되지만 korea 또는 korean 에는 매칭이 안되는 패턴?"
pattern1 = re.compile( r'\bkor\b' )
rst_matchLst1 = re.findall( pattern1, text1 )
print( "[1] : ", rst_matchLst1 )


# [2]
text2 = "(Question) kor 단어는 매칭되지만 korea 또는 korean 에는 매칭이 안되는 매칭 패턴은?"
pattern2 = re.compile( r'\b매칭\w*\b' )  # 매칭되지만, 매칭이, 매칭
rst_matchLst2 = re.findall( pattern2, text2 )
print( "[2] : ", rst_matchLst2 )









