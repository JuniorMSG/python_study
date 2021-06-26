import os

print('')
print('__file__                     :', __file__)

# 심볼릭 링크등의 실제 경로를 찾아준다.
print('os.path.realpath(__file__)   :', os.path.realpath(__file__))

# 파일의 절대 경로를 리턴한다.
print('os.path.abspath(__file__)    :', os.path.abspath(__file__))

# 현재 파일의 디렉토리 경로
print('현재 파일의 디렉토리 경로 :', os.getcwd())

# 현재 파일의 폴더 경로 리턴
print('현재 파일의 폴더 경로 리턴 :', os.path.dirname(os.path.realpath(__file__)))

# 현재 디렉토리에 있는 파일 리스트
print('현재 디렉토리에 있는 파일 리스트 :', os.listdir(os.getcwd()))

# OS별 다른 파일 구분자(file separate) 사용방법
print('OS별 다른 separate 사용방법 :', os.path.sep)

# 작업 디렉토리 변경
print(os.getcwd())
os.chdir('../')
print(os.getcwd())


