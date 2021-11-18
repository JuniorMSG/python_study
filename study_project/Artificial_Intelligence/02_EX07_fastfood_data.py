"""
    subject
        Machine_Running
    topic
        비즈니스 데이터 실습 - 패스트푸드 매장 분포 분석

    Describe

    Contens
        01.
"""

import os
import numpy as np
import pandas as pd

import seaborn as sns

from matplotlib import font_manager, rc
import matplotlib as mpl
import matplotlib.pyplot as plt

import glob
import missingno
# pip install plotnine
# from plotnine import *
import folium #지도시각화


def font_set():
    font_path = "C:\Windows\Fonts\HYGTRE.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font)
    plt.rc('axes', unicode_minus=False)

    plt.rcParams["font.size"] = 14
    plt.rcParams["figure.figsize"] = (12, 6)
    # plt.figure(figsize=(5, 5))
    mpl.rcParams['axes.unicode_minus'] = False

    plt.plot([0, 1], [0, 1], label='한글테스트용')
    plt.legend()
    # plt.show()


"""
    subject
        Machine_Running
    topic
        비즈니스 데이터 실습 - 상가정보로 뭔가 해보기
    Describe
        
    Contens
        01.
"""

import matplotlib.font_manager as fm

# 폰트가 있는지 확인해봅니다.
sys_font=fm.findSystemFonts()
print(f"sys_font number: {len(sys_font)}")
print(sys_font)

# 현재 설정되어 있는 폰트 사이즈와 글꼴을 확인해봅니다.
def current_font():
    print(f"현재 설정된 폰트 글꼴: {plt.rcParams['font.family']}, 현재 설정된 폰트 사이즈: {plt.rcParams['font.size']}")  # 파이썬 3.6 이상 사용가능

font_set()
current_font()

df_se = pd.read_csv('data_file/소상공인시장진흥공단_상가(상권)정보_20210331/소상공인시장진흥공단_상가(상권)정보_서울_202103.csv', encoding='utf-8')
df_gy = pd.read_csv('data_file/소상공인시장진흥공단_상가(상권)정보_20210331/소상공인시장진흥공단_상가(상권)정보_경기_202103.csv', encoding='utf-8')
df_da = pd.read_csv('data_file/소상공인시장진흥공단_상가(상권)정보_20210331/소상공인시장진흥공단_상가(상권)정보_대전_202103.csv', encoding='utf-8')
df_bu = pd.read_csv('data_file/소상공인시장진흥공단_상가(상권)정보_20210331/소상공인시장진흥공단_상가(상권)정보_부산_202103.csv', encoding='utf-8')

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html

# df_store = pd.concat([df_se, df_gy, df_da, df_bu] , axis=0)
# axis=1 - 가로로 합치기
# axis=0 - 세로로 합치기 Default
df_store = pd.concat([df_se, df_gy, df_da, df_bu])
df_store.reset_index(drop=True, inplace=True)
# df_store = df_store.reset_index(drop=True)

# 절대경로 얻기
path = os.path.dirname(os.path.abspath(__file__))
folder_path = path + os.sep + 'data_file' + os.sep + '소상공인시장진흥공단_상가(상권)정보_20210331' + os.sep
folder_img_path = folder_path + 'vis_img' + os.sep
folder_data_concat_path = folder_path + 'data_concat' + os.sep

img_li = os.listdir(folder_img_path)
data_concat_li = os.listdir(folder_data_concat_path)


def data_concat(path, option, savename):
    folder_li = glob.glob(path + option)
    # for문
    for i in range(len(folder_li)):
        # df명
        df = 'df_store' + str(i + 1)
        print(df)

        # 파일 로드
        df = pd.read_csv(folder_li[i], encoding='utf-8', delimiter=',')

        # df명을 넣는 조건
        if i == 0:
            df_store = df
        elif i > 0:
            df_store = pd.concat([df_store, df], axis=0)

    df_store.to_csv(folder_data_concat_path + savename)

    return


data_file_name = '202103 상가정보.csv'
if data_file_name not in data_concat_li:
    data_concat(folder_path, '*.csv', data_file_name)


# 인덱스 재지정
df_store = df_store.reset_index(drop=True)
print(df_store.shape)

# 정보
print(df_store.info())

# 결측치
print(df_store.isnull().sum())


# 결측치 시각화 ( 오래걸림 )
file_name = '01_결측치 그래프.png'
if file_name not in img_li:
    fig = plt.figure(figsize=(16,6))
    sns.heatmap(df_store.isnull(), cbar=False)
    plt.xticks(rotation=45, fontsize=12)
    fig.get_figure().savefig(folder_img_path + file_name)
    plt.show()


"""
    결측치 처리하기
        결측치를 처리하는 방법에는 크게 삭제 혹은 특정값으로 채우는 방법이 있습니다.
        데이터가 많을 경우 데이터를 삭제할 수도 있겠지만, 그렇지 않다면 데이터는 소중하기 때문에 특정값으로 대체하게 됩니다.
        결측치 처리방법은 아래 링크를 참조해주세요.
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.notnull.html
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.any.html
        https://machinelearningmastery.com/handle-missing-data-python/
        https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html
        https://eda-ai-lab.tistory.com/14
        https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null
"""


# 결측값이 비교적 적은 칼럼들만 선정
# df_store.columns
df_clr_columns = ['상가업소번호', '상호명', '상권업종대분류코드',
              '상권업종대분류명', '상권업종중분류코드','상권업종중분류명',
              '상권업종소분류코드', '상권업종소분류명',
              '시도코드','시도명', '시군구코드', '시군구명', '행정동코드',
              '행정동명', '법정동코드', '법정동명', '지번코드','대지구분코드',
              '대지구분명', '지번본번지','지번주소', '도로명코드', '도로명',
              '건물본번지','도로명주소', '신우편번호',
              '경도', '위도']

df_store_clr = df_store[df_clr_columns].copy()
print(df_store_clr.shape)
print(df_store_clr.isnull().sum())

# null값을 포함한 row 출력하기
temp1 = df_store_clr[df_store_clr.isnull().any(axis=1)]
print(temp1.shape)

# 모든 칼럼에 null값이 없는 row들만 출력하기
temp2 = df_store_clr[df_store_clr.notnull().all(axis=1)]
print(temp2.shape)

# null값이 있는 row 제거
df_store_clean = df_store_clr.dropna(axis=0)
print(df_store_clean.shape)

# df_store_clean 결측치 수
print(df_store_clean.shape)
print(df_store_clean.isnull().sum())

# 지리정보(위도,경도)를 이용해 지도 그리기, 위도와 경도로 산점도를 그려봅니다.
# 대한민국 지도
# 데이터 수가 많아 시간이 조금 걸릴 수 있습니다.

file_name = '02_산점도 그래프 위도, 경도.png'
if file_name not in img_li:
    fig = df_store_clean.plot.scatter(x='경도', y='위도', figsize=(12,8), grid=True)
    fig.get_figure().savefig(folder_img_path + file_name)
    plt.show()


# df_store_clean 의 '도로명주소' 칼럼을 10행만 출력해봅니다.
df_store_clean['도로명주소'][:10]

# '도로명주소'를 기준으로 서울과 그 외 지역으로 구분해봅니다.
df_store_seoul = df_store_clean.loc[df_store_clean['도로명주소'].str.startswith('서울')]
df_store_other = df_store_clean.loc[~df_store_clean['도로명주소'].str.startswith('서울')]

# matplotlib으로 시각화
# 서울
file_name = '03_서울 산점도 그래프.png'
if file_name not in img_li:
    fig = df_store_seoul.plot.scatter(x='경도',y='위도',figsize=(12,8), grid=True)
    fig.get_figure().savefig(folder_img_path + file_name)
    plt.show()

# 다른 지역 전체
file_name = '04_다른지역 전체.png'
if file_name not in img_li:
    fig = df_store_other.plot.scatter(x='경도',y='위도',figsize=(12,8), grid=True)
    fig.get_figure().savefig(folder_img_path + file_name)
    plt.show()


# '시군구명' 칼럼
print(df_store_seoul['시군구명'][:10])


# seaborn으로 시각화
# 서울을 시군구명으로 구분지어 표시해봅니다.
file_name = '05_서울을 시군구명.png'
if file_name not in img_li:
    fig = plt.figure(figsize=(18, 10))
    ax = sns.scatterplot(data=df_store_seoul,
                         x='경도',
                         y='위도',
                         hue='시군구명')
    plt.setp(ax.get_legend().get_texts(), fontsize='10')
    plt.savefig(folder_img_path + file_name)
    plt.show()

# 인코딩 관련
"""
    인코딩 관련
        https://docs.python.org/ko/3.8/howto/unicode.html
        https://croak.tistory.com/44
        https://studyforus.tistory.com/167
        https://ifyourfriendishacker.tistory.com/5
"""
df_store_21 = df_store[df_store['상호명'].notnull()]
df_store_21_clean = df_store[df_store['상호명'].notnull()]

temp1 = df_store_21_clean[df_store_21_clean['상호명'].str.contains('맥도날드')]
temp2 = df_store_21_clean[df_store_21_clean['상호명'].str.lower().str.contains('맥도날드|mcdonald')]
print(temp2['상권업종소분류명'].unique())

"""
    필요한 패스트푸드점의 데이터를 선별합니다.
    조건
    '도로명주소'가 '서울'로 시작하는 경우
    '상권업종소분류명'이 '패스트푸드' 인 경우
    '상호명'에 브랜드명이 포함되는 경우
"""

df_mac_21 = df_store_21_clean[
    (df_store_21_clean['도로명주소'].str.startswith('서울')) &
    (df_store_21_clean['상권업종소분류명'] == '패스트푸드') &
    (df_store_21_clean['상호명'].str.lower().str.contains('맥도날드|mcdonald'))]

df_mac_21['year'] = '2021'
df_mac_21['brand'] = 'mac'

df_bk_21 = df_store_21_clean[
    (df_store_21_clean['도로명주소'].str.startswith('서울')) &
    (df_store_21_clean['상권업종소분류명'] == '패스트푸드') &
    (df_store_21_clean['상호명'].str.lower().str.contains('버거킹|burgerking'))]

df_bk_21['year'] = '2021'
df_bk_21['brand'] = 'bk'


df_moms_21 = df_store_21_clean[
    (df_store_21_clean['도로명주소'].str.startswith('서울')) &
    (df_store_21_clean['상권업종소분류명'] == '패스트푸드') &
    (df_store_21_clean['상호명'].str.lower().str.contains('맘스터치|momstouch'))]

df_moms_21['year'] = '2021'
df_moms_21['brand'] = 'moms'

df_lott_21 = df_store_21_clean[
    (df_store_21_clean['도로명주소'].str.startswith('서울')) &
    (df_store_21_clean['상권업종소분류명'] == '패스트푸드') &
    (df_store_21_clean['상호명'].str.lower().str.contains('롯데리아|lotteria'))]

df_lott_21['year'] = '2021'
df_lott_21['brand'] = 'lott'

df_fast_food_21 = pd.concat([df_mac_21, df_moms_21, df_lott_21, df_bk_21])
print(df_fast_food_21.isnull().sum(0))

df_fastfood_year = pd.DataFrame(df_fast_food_21.groupby(by=['year','brand'])['brand'].count())
df_fastfood_year.columns = ['매장수']
print(df_fastfood_year)

# 멀티 인덱스 중 원하는 인덱스를 칼럼으로 보낼때 reset_index를 사용합니다.
df_fastfood_year.reset_index(level=["brand"], inplace=True)
print(df_fastfood_year)

# barplot
# 다른 브랜드들과 비교

# plot 사이즈 조절
plt.figure(figsize=(12,8)) # 차트 사이즈
plt.rc('legend', fontsize=14) # 범례 사이즈
plt.rc('axes', labelsize=20) # 축 제목
plt.rc('font', size=20) # x,y 축 값
# plt.xticks(fontsize=20, rotation=0) # x축 값

sns.barplot(data = df_fastfood_year,
           x = df_fastfood_year.index,
           y = '매장수',
           hue = 'brand',
           palette="Blues_d"
           )

plt.show()


# lineplot
# 다른 브랜드들과 비교

# plot 사이즈 조절
plt.figure(figsize=(12,8)) # 차트 사이즈
plt.rc('legend', fontsize=14) # 범례 사이즈
plt.rc('axes', labelsize=20) # 축 제목
plt.rc('font', size=20) # x,y 축 값
# plt.xticks(fontsize=20, rotation=0) # x축 값

palette = sns.color_palette("mako_r", 4)

sns.lineplot(data = df_fastfood_year,
           x = df_fastfood_year.index,
           y = '매장수',
           hue = 'brand',
           style="brand",
           palette=palette,
           markers=True,
           sizes=(6,6)
           )

plt.show()

# 사용할 데이터
df_geo = df_fast_food_21

# 지도를 처음 열어줄 때, 어디를 중심으로 보여줄지 설정합니다.
# 위도와 경도의 평균값을 중심점으로 잡습니다.
# zoom_start는 처음 지도의 크기를 뜻합니다.
map = folium.Map(location=[df_geo['위도'].mean(),df_geo['경도'].mean()], zoom_start=11.5)

# 지도 시각화
for g in df_geo.index:

    if (df_geo.loc[g, 'brand'] == 'mac') & (df_geo.loc[g, 'year'] == '2021'):
        icon_color = 'red'
        icon_shape = 'home'
    elif (df_geo.loc[g, 'brand'] == 'bk') & (df_geo.loc[g, 'year'] == '2021'):
        icon_color = 'blue'
        icon_shape = 'home'
    elif (df_geo.loc[g, 'brand'] == 'lott') & (df_geo.loc[g, 'year'] == '2021'):
        icon_color = 'green'
        icon_shape = 'home'
    elif (df_geo.loc[g, 'brand'] == 'moms') & (df_geo.loc[g, 'year'] == '2021'):
        icon_color = 'orange'
        icon_shape = 'home'

    # https://python-visualization.github.io/folium/
    # https://getbootstrap.com/docs/3.3/components/
    # https://python-visualization.github.io/folium/modules.html#folium.map.Icon
    # folium.Map?
    folium.Marker([df_geo.loc[g, '위도'], df_geo.loc[g, '경도']], icon=folium.Icon(color=icon_color, icon=icon_shape)).add_to(map)

map.save(folder_img_path + 'geo_01.html')
"""
    https://python-visualization.github.io/folium/quickstart.html#Choropleth-maps
    https://hashcode.co.kr/questions/1772/%EB%A7%88%ED%81%AC%EB%8B%A4%EC%9A%B4-%EB%AC%B8%EB%B2%95-%EC%9E%91%EC%84%B1-%ED%8C%81
    https://www.python-graph-gallery.com/
    https://www.data.go.kr/
    https://data.seoul.go.kr/
    https://bigdata.seoul.go.kr/main.do
    https://www.kbfg.com/kbresearch/report/reportView.do?reportId=1003860
"""