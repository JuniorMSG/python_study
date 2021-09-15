# \b와 \B를 이용한 문자열 검색 패턴
# 앞에서 배운 \b, \B 에 대한 문제가 조금 쉬었다해서 이 둘을 만만하게 보면 안된다. 괜히 사람들이 많이 헤갈려하는게 아니다.
# 이제부터의 문제들을 통해서 많이 헤갈리는 \b, \B 를 잘 이해해보자.
# 이 소스코드 페이지에서의 강의목적은 크게 2가지 --> (1) 메타문자 쓰임새 알기    (2) \b, \B 정확히 이해하고 써먹기
import re


# [1] :    [ ]
# (Question) kor 단어는 매칭이 되지만, korea 또는 korean 단어에는 매칭이 안되는 패턴을 만드시오?
text1 = "(Question) kor 단어는 매칭이 되지만, Korea 또는 korean 단어에는 매칭이 안되는 패턴을 만드시오?"
pattern1 = re.compile( r'[a-zA-Z]' )  # o, k, o, r, k, o, r, k, o, r
rst_matchLst1 = re.findall( pattern1, text1 )
print( "[1] : ", rst_matchLst1 )

# [ ] : 대괄호는 문자 패턴의 집합을 나타냄.
# or (또는)의 개념으로 생각하면 쉽다.
# [abc] ---> a 또는 b 또는 c 또는 의 의미로써 3개(a, b, c) 문자중 하나가 있으면 매칭된다.
# [a-zA-Z] ---> 마이너스(-) 기호를 사용하여 범위를 지정해 줄 수 있다. a 부터 z 까지중 하나가 있다면 매칭.
# 정규표현식에서 가장 많이 쓰이는 것중에 하나이다. 대괄호 안에는 하나 또는 하나 이상의 문자들이 올 수 있다.


# [2]
text2 = "(Question) kor 단어는 매칭이 되지만, Korea 또는 korean 단어에는 매칭이 안되는 패턴을 만드시오? 123, 456, 789"
pattern2 = re.compile( r'[a-zA-Z0-9]' )  # 소문자, 대문자, 숫자
rst_matchLst2 = re.findall( pattern2, text2 )
print( "[2] : ", rst_matchLst2 )


# [3] :    .
text3 = "(Question) kor 단어는 매칭이 되지만, Korea 또는 korean 단어에는 매칭이 안되는 패턴을 만드시오?"
pattern3 = re.compile( r'.{3}' )
rst_matchLst3 = re.findall( pattern3, text3 )
print( "[3] : ", rst_matchLst3 )


# [4]
# 주어진 텍스트에서 영문자(단어)만 검색하고자 한다면?
text4 = "(Question) kor 단어는 매칭이 되지만, Korea 또는 korean 단어에는 매칭이 안되는 패턴을 만드시오?"
pattern4 = re.compile( r'[a-zA-Z]+' )  # + : 1개 이상
rst_matchLst4 = re.findall( pattern4, text4 )
print( "[4] : ", rst_matchLst4 )


# [5]
text5 = "(Question) kor 단어는 매칭이 되지만, Korea 또는 korean 단어에는 매칭이 안되는 패턴을 만드시오?"
pattern5 = re.compile( r'[a-zA-Z]*' )  # * : 0개 이상 
rst_matchLst5 = re.findall( pattern5, text5 )
print( "[5] : ", rst_matchLst5 )


# [6]
# 주어진 텍스트에서 korea, korean 단어들만 검색하시오?
# 주어진 텍스트에서 kor, korea, korean 단어들만 검색하시오?
text6 = "(Question) kor 단어는 매칭이 되지만, korea 또는 korean 단어에는 매칭이 안되는 패턴을 만드시오?"
pattern6 = re.compile( r'korean|korea|kor' )
rst_matchLst6 = re.findall( pattern6, text6 )
print( "[6] : ", rst_matchLst6 )


# [7]
# 주어진 텍스트에서 kor 단어만 검색하시오?
text7 = "(Question) kor 단어는 매칭이 되지만, korea 또는 korean 단어에는 매칭이 안되는 패턴을 만드시오? ork rko okr"
pattern7 = re.compile( r'[kor]{3}' )
rst_matchLst7 = re.findall( pattern7, text7 )
print( "[7] : ", rst_matchLst7 )






























