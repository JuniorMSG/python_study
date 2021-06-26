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
