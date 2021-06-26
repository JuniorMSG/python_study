import os

# 상대경로로 파일 호출하는 방법
file_path = './etc_file_data' + os.path.sep + 'text.txt'
file = open(file_path, 'w', encoding='CP949')

for i in range(1, 5):
    data = "%d번째 줄입니다.\n" % i
    file.write(data)

file = open(file_path, 'r', encoding='CP949')
print(file.readline())
print(file.readlines())
file.close()

# 절대경로로 파일 호출하는 방법
folder = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'etc_file_data'
file_nm = 'text.txt'

file_path = folder + os.path.sep + file_nm
print('절대 경로 :', file_path)

file = open(file_path, 'w', encoding='CP949')

for i in range(1, 11):
    data = "%d번째 줄입니다.\n" % i
    file.write(data)

print(file)
file.close()

file = open(file_path, 'r', encoding='CP949')
print(file.readline())
print(file.readlines())
file.close()



# 작업 디렉토리 변경
print(os.getcwd())
os.chdir('../')
print(os.getcwd())


