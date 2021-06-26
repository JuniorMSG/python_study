"""
Author  : SG
DATE    : 2021-05-27

03. 제어문 , 반복문

03_01. if       (제어문)
03_02. while    (반복문)
03_03. for      (반복문)
★ 03_04. Question algorithm

★ 03_04. Question algorithm
git@gist.github.com:5e7e43aa4f37db1daab63d5e11f2c54c.git
"""


# 소수 찾기 공식
sosu = []
for x in range(2, 9999):
    for y in range(2, x+1):
        if x == y:
            sosu.append(x)
            break
        if x % y == 0:
            break

print(sosu)
sosubin = []
sosucnt = []

for data in sosu:
    cnt = 0
    for x in bin(data):
        if x == "0":
            cnt += 1

    sosubin.append(bin(data))
    sosucnt.append(cnt-1)


print(sosubin)
print(sosucnt)
a = int("0b1001101", 2)
print(a)