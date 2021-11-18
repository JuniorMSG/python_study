from bs4 import BeautifulSoup as BS
import requests as req

"""
    Crawling_04_selector
    
    xpath 란 
    xml 문서에서 Element 들과 Attribute 들을 탐색하기 위한 selector
    / 해당 노드 바로 안의
    // 해당 노드안의 안의 상관없이
    . 현재노드
    .. 부모노드 
    @ : attribute
    
    css ( Cascading STyle Sheets
    html 이 구조 잡은 곳에 스타일링(색칠, 크기 등)하는 것
    
    스타일링에 스타일 이름을 붙여 (class)
    구조에 스타일 이름을 넣음
    
    스타일 이름이 없는 구조에도 스타일링을 함 (css selector)
    
    구조는 스타일 이름으로 특정 지어질 수 있음.
    Element Type 방식
        document.querySelecotr("div")
        document.querySelectorAll("div")
        
        tag = div, span
        class = .sale
        id = #market
        
        조합 
        document.querySelector("div.tbl_area")
        
    ID 방식       
    Class 방식  
        HTML Element의 그룹임  
        
        
    고급 한정자 방식
        # 태그 관련 정보 
        https://w3schools.com/tags/ref_attributes.asp
        
        =  : 일치한다.
        *= : 포함한다.
        ^= : ~으로 시작한다.
        $= : ~으로 끝난다.
        
        
        * 모든 노드들
        div, p div 와 p 노드들
        div p div 안에 있는 p 노드들
        div > p div 바로 안에 있는 p 노드들
        div ~ p p옆(앞)에 있는 div 노드들
        div + p div 옆(뒤)에 있는 p 노드들 
        
        :enabled        : 활성화된 상태
        :checked        : 체크 딘 상태
        :disabled       : 비활성화 된 상태
        :empty          : 값이 비어 있는 상태 
        :first-child    : 첫번째 자식
        :last_child     : 마지막 자식
        :first-of-type  : 해당 타입의 첫번째 자식 노드
        :last-of-type   : 해당 타입의 마지막 노드
        :hover          : 마우스가 올라간 상태
        :not            : 다음 조건이 거짓일 경우
        :nth-child      : n 번째 자식
        :nth-of-type    : n 번째 타입 
        
        div     :
        p       : paragraph  ( p단위로 엔터가 들어감.)
        span    :
        pre     :

    자동생성된 것 같은 class명 피하기.
"""

url = 'https://search.shopping.naver.com/search/all?query=%EC%97%90%EC%96%B4%ED%8C%9F&cat_id=&frm=NVSHATC'
res = req.get(url)
soup = BS(res.text, 'html.parser')

arr = soup.select('ul.list_basis div>a:first-child[title]')

# 정적 크롤링은 한계가 있음.
# 무한스크롤 적용 안되고, 데이터를 html 형식으로 안가지고 있을 수 있음.
for a in arr:
    print(a.get_text(strip=True))

url = 'https://www.coupang.com/np/search?component=&q=%EB%85%B8%ED%8A%B8%EB%B6%81&channel=userC'
res = req.get(url, headers={
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
})
soup = BS(res.text, 'html.parser')

# list comprehension
arr = [a for a in range(5)]
print(arr)
# syntax sugar 파이썬 다운 문법
# arr = [a.get_text(strip=True) for a in soup.select('div.name')]
# print(arr)

arr = soup.select('div.name')
for a in arr:
    print(a.get_text(strip=True))

for desc in soup.select("div.descriptions-inner"):
    ads = desc.select("span.ad-badge")
    if len(ads) > 0:
        print("광고")
    print(desc.select("div.name")[0].get_text(strip=True))




url = 'https://finance.naver.com/sise/lastsearch2.nhn'
res = req.get(url, headers={
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
})
soup = BS(res.text, 'html.parser')

tr_lst = soup.select('table.type_5 tr')

for tr in tr_lst:
    if len(tr.select("a.tltle")) == 0:
        continue
    title = tr.select('a.tltle')[0].get_text(strip=True)
    price = tr.select('td.number:nth-child(4)')[0].get_text(strip=True)
    change = tr.select('td.number:nth-child(6)')[0].get_text(strip=True)
    print(title, ":", price, change)

url = "https://finance.yahoo.com/most-active"
res = req.get(url, headers={
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
})
soup = BS(res.text, 'html.parser')


for a in soup.select("table tbody tr"):
    title = a.select("td:nth-child(1) a")[0].get_text(strip=True)
    price = a.select("td:nth-child(3)")[0].get_text(strip=True)
    change = a.select("td:nth-child(5) span")[0].get_text(strip=True)
    print(title, price, change)
