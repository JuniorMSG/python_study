"""
    파이썬 re 모듈

    match() 함수는 기본적으로 원본 문장(문자열)의 시작부터  패턴 매칭을 검색한다.
    처음에 없으면 더 이상 찾지 않는다. 설령 중간이나 끝에 패턴 매칭이 되는게 있다 하더라도.
    따라서 패턴과 일치되는게 처음에 없으면 없는거다.
    검색 결과에서 다른 정보 필요없이 값만 출력하고자 한다면 group() 메서드를 사용한다.
"""
import re

pattern = "kor"
sentence1 = "I am a korean."
sentence2 = "korean boy"
sentence3 = "Korean boy"
sentence4 = "korea vs korean의 차이는?"

# [ 1 ] 검색 패턴을 미리 컴파일
# 이렇게하면 아무래도 여러 번 사용하는 경우 매번 패턴을 또 지정안해서 좋음.
# 검색 속도도 좀 더 빠름.
# 같은 검색을 여러 번 사용하는 경우라면 좋음.
sp = re.compile(pattern)
rst_matchObj_1 = sp.match(sentence1)  # 매치결과는 matchObject 인스턴스 객체 반환
rst_matchObj_2 = sp.match(sentence2)
rst_matchObj_3 = sp.match(sentence3)
rst_matchObj_4 = sp.match(sentence4)

print("1번 문장의 패턴 매칭 결과는 : {} 입니다.".format(rst_matchObj_1))
print("2번 문장의 패턴 매칭 결과는 : {} 입니다.".format(rst_matchObj_2))
print("3번 문장의 패턴 매칭 결과는 : {} 입니다.".format(rst_matchObj_3))
print("4번 문장의 패턴 매칭 결과는 : {} 입니다.".format(rst_matchObj_4))


# 위 검색 결과를 보면 여러 매칭된 결과 정보까지 함께 출력됨을 알 수 있다.
# matchObject 객체로부터 매칭된 값에 대한 정보만 원한다면 group() 메서드를 사용한다.
rst_matchObj_2 = sp.match(sentence2).group()
print( "2번 문장의 패턴 매칭 값은 : {} 입니다.".format(rst_matchObj_2) )


# group 메서드와 인덱스를 사용하여 각 그룹별 문자열 반환하기
# 문자열(텍스트)
text = "상품구매와 관련한 사항은 010-1234-4567 번호로 문의주세요!"

# 검색 패턴 매치 - 전화번호만 발췌
sp = re.compile( r"(\d\d\d)-(\d\d\d)-(\d\d\d\d)" )
#sp = re.compile( r"(\d+)-(\d+)-(\d+)" )
#sp = re.compile( r"\d+[^\w]\d+[^\w]\d+" )
rst_matchObj = sp.search(text)

# 출력
print( rst_matchObj )
"""
print( rst_matchObj.group(0) )  # group(0) : 매치된 전체 문자열
print( rst_matchObj.group(1) )  # group(1) : 첫 번째 그룹에 해당하는 문자열 반환
print( rst_matchObj.group(2) )  # group(2) : 두 번째 그룹에 해당하는 문자열 반환
print( rst_matchObj.group(3) )  # group(3) : 세 번째 그룹에 해당하는 문자열 반환
"""


# 1번째인자: pattern, 2번째인자: string(text), 3번째인자: 옵션
rst_matchObj = re.match( 'kor', 'Korea' )

# 출력
print( rst_matchObj )


