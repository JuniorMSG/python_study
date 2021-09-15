# re 모듈을 이용한 정규식의 기초 - backslash
# 어떤 문자열(텍스트) 내용에서 \superman 이라는 문자열을 검색하고자 한다면???
# 또는 \s 라는 문자열을 검색하고자 한다면???
# 이럴 때 생각보다 상황이 복잡해진다.
import re

# [1]
# print문 통해서 \ 출력
text1 = "\superman"
print( "[1] : ", text1 )

# [2]
text2 = r"\\\\\\\superman"
print( "[2] : ", text2 )

# [3]
text3 = '\\\superman'
pattern3 = re.compile( r'\\\\superman' )  # 파이썬 정규식 엔진 내부에서 문자열 처리 규칙에 따라 \\(2개를) -> \(1개로) 컴파일 함.
rst3 = re.search( pattern3, text3 )
print( "[3] : ", rst3 )

# [4]
text4 = r"superman 단어 앞에다 백슬래시를 붙이면 \\\\\\superman 이렇게 됩니다."
pattern4 = re.compile( r'\\\\\\\\\\\\superman' )
rst4 = re.search( pattern4, text4 )
print( "[4] : ", rst4.group() )

# [5]
text5 = r"superman 단어 앞에다 백슬래시를 붙이면 \\\superman 이렇게 됩니다."
pattern5 = re.compile( r'\\\\\\s' )
rst5 = re.search( pattern5, text5 )
print( "[5] : ", rst5.group() )



















