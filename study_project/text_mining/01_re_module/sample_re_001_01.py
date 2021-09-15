# group 메서드와 인덱스를 사용하여 각 그룹별 문자열 반환하기
import re

# 문자열(텍스트)
text = "상품구매와 관련한 사항은 010-1234-4567 번호로 문의주세요!"

# 검색 패턴 매치 - 전화번호만 발췌
sp = re.compile( r"(\d\d\d)-(\d\d\d)-(\d\d\d\d)" )
#sp = re.compile( r"(\d+)-(\d+)-(\d+)" )
#sp = re.compile( r"\d+[^\w]\d+[^\w]\d+" )
rst_matchObj = sp.search( text )

# 출력
print( rst_matchObj )
"""
print( rst_matchObj.group(0) )  # group(0) : 매치된 전체 문자열
print( rst_matchObj.group(1) )  # group(1) : 첫 번째 그룹에 해당하는 문자열 반환
print( rst_matchObj.group(2) )  # group(2) : 두 번째 그룹에 해당하는 문자열 반환
print( rst_matchObj.group(3) )  # group(3) : 세 번째 그룹에 해당하는 문자열 반환
"""





