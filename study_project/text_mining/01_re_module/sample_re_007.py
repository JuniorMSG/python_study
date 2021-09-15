# 주어진 텍스트내에서 영문자가 3개 또는 5개로 구성된거 다 찾기?
import re                               


# [1]
text1 = "(Question) kor 단어는 검색되지만 korea 또는 korean 에는 매칭이 안되는 패턴? a ab abc abcd abcde abcdef"
pattern1 = re.compile( r'\b[a-zA-Z]{3}\b|\b[a-zA-Z]{5}\b' )
rst_matchLst1 = re.findall( pattern1, text1 )
print( "[1] : ", rst_matchLst1 )


# [2] : 영어 단어만 검색하려면?
text2 = "Question : kor1 단어는 검색되지만 korea 또는 korean 에는 매칭이 안되는 패턴? a ab abc abcd abcde abcdef"
pattern2 = re.compile( r'[a-zA-Z]+' )
#text2 = "Question15412478abcdef"
#pattern2 = re.compile( r'^[a-zA-Z0-9]+$' )  # 처음과 끝을 표시
rst_matchLst2 = re.findall( pattern2, text2 )
print( "[2] : ", rst_matchLst2 )


# [3] : 영문 세글자 이상만 검색하려면?
text3 = "Question : kor 단어는 검색되지만 korea 또는 korean 에는 매칭이 안되는 패턴? a ab abc abcd abcde abcdef"
pattern3  = re.compile( r'[a-zA-Z]{3,}' )
rst_matchLst3 = re.findall( pattern3, text3 )
print( "[3] : ", rst_matchLst3 )


# [4] : 영문 소문자 세글자 이상짜리 단어만 검색하려면?
text4 = "Question : kor 단어는 검색되지만 korea 또는 korean 에는 매칭이 안되는 패턴? a ab abc abcd abcde abcdef"
pattern4 = re.compile( r'\b[a-z]{3,}\b' )
rst_matchLst4 = re.findall( pattern4, text4 )
print( "[4] : ", rst_matchLst4 )


# [5] : caaaat 에서 aaaa만 매칭시키고 싶다면?
text5 = "ccccaaaabbbb"
pattern5 = re.compile( r'\Baaaa\B' )
rst_matchLst5 = re.findall( pattern5, text5 )
print( "[5] : ", rst_matchLst5 )


# [6] : [5]번 복습
# JupyterNotebook 에는 매치되지만, evernote, notebook, onenote 등에는 매치되지 않는 정규식은?
text6 = "JupyterNotebook"
pattern6 = re.compile( r'\Bnote\B', re.IGNORECASE )
rst_matchLst6 = re.findall( pattern6, text6 )
print( "[6] : ", rst_matchLst6 )


# [7]
# 아래 텍스트에서 ‘Notebook’이 들어간(포함된) 단어(문자열)를 모두 매치하려면?
text7 = '''
jupyternotebook, evernote, notebook, onenote, notebook1, 21세기notebook, jupyter-notebook, 
jupyter_notebook, jupyter~notebook, jupyter1notebook
'''
#pattern7 = re.compile( r'[a-zA-Z]*notebook\w*' )
#pattern7 = re.compile( r'\w*[-@]*notebook\w*' )  # [-@] 도 괜찮으나 어떤게 또 있을지 모르므로 약간의 수정 필요.
pattern7 = re.compile( r'\w*[^\w\s]*notebook\w*' )  # [ ] 안의 ^ 는 부정의 의미.
rst_matchLst7 = re.findall( pattern7, text7 )
print( "[7] : ", rst_matchLst7 ) 


# [8] : [^ab] 용법
text8 = "abc-def-123-456"
pattern8 = re.compile( r'[^ab]' )  ################(1) : 이것과 같음 --> [^a^b], [^(ab)]
#pattern8 = re.compile( r'^ab' )  ##################(2) : [ ] 빼면 결과가 완전 달라짐. ab로 시작하는 걸 찾음.
rst_matchLst8 = re.findall( pattern8, text8 )
print( "[8] : ", rst_matchLst8 )




























