from bs4 import BeautifulSoup as BS
import requests as req

"""
    Crawling_04_BeautifulSoup
    
    html 문자열 파싱
    html 노드 인식 및 편리한 기능들
    parent, children, contents, descendants, sibling
    string, strings, stripped_strings, get_text()
    prettify (uglify - 난독화)
        - 이쁘게 코드를 받아옴?
    html attribute
"""

url = 'https://finance.naver.com/marketindex/exchangeList.naver'
res = req.get(url)
soup = BS(res.text, 'html.parser')

print(soup.title)

tds = soup.find_all('td')

names = []
for td in tds:
    if len(td.find_all("a")) == 0:
        continue
    # 통화명
    print(td.get_text(strip=True))
    names.append(td.get_text(strip=True))

prices = []
for td in tds:
    if 'class' in td.attrs:
        if "sale" in td.attrs["class"]:
            prices.append(td.get_text(strip=True))

print(names)
print(prices)
