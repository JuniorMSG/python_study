"""
04. pass
파이썬은 조건문 내부에 아무것도 작성하지 않으면 오류가 나는데
아무런 일도 하지 않도록 설정할때 사용하게 됩니다.
"""

print("\n", "=" * 5, "04. pass", "=" * 5)

data = [1, 2, 3]
if 3 in data:
    pass
else:
    print(data)
