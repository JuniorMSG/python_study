# 정규식 사용시 많이 헤갈리는 특수문자 \b, \B
# b의 사전적의미 : 경계를 의미하는 border, boundary
# 즉, 단어와 단어 사이의 경계 지점을 의미. (각 단어의 경계) == 문자와 문자 사이의 경계 지점.
# 한마디로 경계의 '위치'를 가리킨다. 그래서 패턴이 일치해도 매칭되는 길이는 0이다.
# r 선언을 해주고 한다. --> 안그러면 결과가 예상하고 다르게 나올 수 있다.
import re


# [1]
# 각각의 결과를 예상하시오?
rst_list1 = re.findall( r'\b', 'Welcome to Seoul.' )
rst_list2 = re.findall( r'\B', 'Welcome to Seoul.' )

# [ 복습 ]
# \b : 단어(문자)의 경계를 의미 --> 즉, 단어(문자)의 앞(시작) 또는 끝을 의미.
# \B : \b(단어문자의 경계)가 아닌 곳을 의미 --> 즉, 비단어문자의 경계를 의미.

# (문제 1) \b의 결과는?           __6__개
# (문제 2) \B의 결과는?           __12__개

# 출력
print( "rst_list1 : ", rst_list1 )
print( "rst_list1 갯수 : ", len(rst_list1), "개" )
print( "--------------------------------------------------" )
print( "rst_list2 : ", rst_list2 )
print( "rst_list2 갯수 : ", len(rst_list2), "개" )
print( "--------------------------------------------------" )


# [2]
print( re.findall( r'\B', 'Every single Koean should be wearing a mask when...' ) )
print( len(re.findall( r'\B', 'Every single Koean should be wearing a mask when...' )) )  # __34__개
print( len(re.findall( r'\b', 'Every single Koean should be wearing a mask when...' )) )  # __18__개


# [3]
print( re.findall( r'\B', '나는 대한민국의 20대 청년입니다.' ) )
print( len(re.findall( r'\B', '나는 대한민국의 20대 청년입니다.' )) )  # __12__개
print( len(re.findall( r'\b', '나는 대한민국의 20대 청년입니다.' )) )  # __8__개


















