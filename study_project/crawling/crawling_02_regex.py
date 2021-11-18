import requests as req
import re

"""
    Crawling_02_정규식 
"""


url = 'https://finance.naver.com/marketindex/?tabSel=exchange#tab_section'
res = req.get(url)
body = res.text

r = re.compile(r"미국 USD.*?value\">(.*?)</", re.DOTALL)
captures = r.findall(body)
print(captures)


r = re.compile(r"h_lst.*?blind\">(.*?)</span>.*?value\">(.*?)</", re.DOTALL)
captures = r.findall(body)
print(captures)


print("===========")
print("=====환율 계산기======")
print("===========")


for c in captures:
    print(c[0], c[1])

print()
usd = float(captures[0][1].replace(",", ""))
won = input("달러로 바꾸길 원하는 금액을 입력")
won = int(won)
dollor = won / usd
print(round(dollor, 2), "$")

