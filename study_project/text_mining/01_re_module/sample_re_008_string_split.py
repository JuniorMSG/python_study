# 특정 단어가 들어간 문자열을 찾을 때 정규식과 문자열 함수 처리중 어떻게 처리하는게 좋을까?
import re


# [1] : 앞서 배웠던 [7]번 예제
# 아래 텍스트에서 ‘Notebook’이 들어간(포함된) 단어(문자열)를 모두 매치하려면?
text7 = '''
jupyternotebook, evernote, notebook, onenote, notebook1, 21세기notebook, jupyter-notebook, 
jupyter_notebook, jupyter@notebook
'''
pattern7 = re.compile( r'[a-zA-Z]*notebook\w*' )
pattern7 = re.compile( r'\w*[-@]*notebook\w*' )
rst_matchLst7 = re.findall( pattern7, text7 )
print( "[7] : ", rst_matchLst7 ) 


# [2]
# 위 text7 텍스트에서 ‘Notebook’이 들어간 단어를 문자열 함수로 처리하려면?
# Python Shell 모드에서 해본다.


# [3]
# 위 text7 텍스트에서 ‘Notebook’이 들어간 단어를 별도의 배열 변수에 저장하여 출력하시오?
# Python Shell 모드에서 해본다.

















