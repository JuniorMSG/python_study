"""
    02. while문
        파이썬에선 강화 for문을 제외한 기본 for문이 없기때문에
        기본 for문은 while문으로 만들어 줘야한다.

        02_01. 기본 구조
            while <조건문>:
            수행할 코드
            수행할 코드
            수행할 코드
        02_02. break
"""

print("\n", "=" * 5, "02. while문", "=" * 5)
print("\n", "=" * 3, "02_01. 기본 구조", "=" * 3)
cnt = 0
while cnt <= 5:
    cnt += 1
    print("Count ", cnt)

print("\n", "=" * 3, "02_02. break", "=" * 3)
breakCnt = 0
while breakCnt <= 5:
    breakCnt += 1
    print("breakCnt ", breakCnt)
    if breakCnt == 3:
        break
"""
    02. 진행 및 종료에 관한 구문
        02_01. 진행 : continue
        02_02. 종료 : break
"""

print("\n", "=" * 5, "02. 진행 및 종료에 관한 구문", "=" * 5)

# 3미출력 5에서 종료
for data in range(10):
    # 02_01 아무것도 하지 않고 다음으로 진행
    if data == 3:
        print("\n", "=" * 3, "02_01. 진행 : continue", "=" * 3)
        continue

    # 02_02. 종료 : break
    if data == 5:
        print("\n", "=" * 3, "02_02. 종료 : break", "=" * 3)
        break
    print(data)