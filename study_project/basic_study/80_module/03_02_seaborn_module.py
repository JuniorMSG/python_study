"""
    subject
        Data analysis module - visualization library
    topic
        seaborn 모듈
    Describe
        matplotlib을 기반으로 다양한 색상과 차트를 지원하는 라이브러리
        matplotlib.pyplot
        import seaborn as sns

        pandas도 matplotlib를 내장하고 있다.

        장점
            1. 아름다운 디자인
            2. 통계 기능 기반의 차트 (countplot, relplot, Implot...)
            3. 쉬운 사용성
            4. pandas, matplotlib 호환



        seaborn doc : https://seaborn.pydata.org/
        pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

    Contents
        font_set : 폰트 설정 함수
        01. why seaborn?
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import numpy as np
import seaborn as sns
import os


def font_set():
    font_path = "C:\Windows\Fonts\HYGTRE.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font)
    # RuntimeWarning: Glyph 8722 missing from current font.
    plt.rc('axes', unicode_minus=False)


def data_set():
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))

    # 데이터 가공
    df_01 = pd.read_csv(path + '/file_data/주택도시보증공사_전국 신규 민간아파트 분양가격 동향_20200430.csv', delimiter=',', encoding='CP949')
    df_01 = df_01.rename(columns={'분양가격(㎡)': '분양가격', '규모구분': '규모', '지역명': '지역'})

    df_01['분양가격'] = df_01['분양가격'].str.strip()
    df_01.loc[df_01['분양가격'] == '', '분양가격'] = 0

    # NAN 0으로 변경
    df_01['분양가격'] = df_01['분양가격'].fillna(0)
    print(df_01.loc[df_01['분양가격'] == '  '])

    # 타입 변경
    df_01['분양가격'] = df_01['분양가격'].astype(int)

    return df_01


def seaborn_01():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            01. Scatterplot (산점도 그래프 )
        Describe
             산점도(散點圖,scatter plot, scatterplot, scatter graph, scatter chart, scattergram, scatter diagram)는
            직교 좌표계(도표)를 이용해 좌표상의 점(點)들을 표시함으로써 두 개 변수 간의 관계를 나타내는 그래프 방법이다.
            도표 위에 두 변수 엑스(X)와 와이(Y) 값이 만나는 지점을 표시한 그림. 이 그림을 통해 두 변수 사이의 관계를 알 수 있다.
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "01. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    x = np.random.rand(50)
    y = np.random.rand(50)
    colors = np.arange(50)
    area = x * y * 1000
    plt.scatter(x, y, s=area, c=colors)
    plt.show()

    sns.scatterplot(x, y, size=area, sizes=(area.min(), area.max()), hue=area, palette='coolwarm')
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    arr = np.random.standard_normal((8, 50))
    plt.subplot(131)
    plt.title('alpha=0.1')
    sns.scatterplot(x, y, size=area, sizes=(area.min(), area.max()), hue=area, palette='coolwarm')

    plt.subplot(132)
    plt.title('alpha=0.5')
    sns.scatterplot(x, y, size=area, sizes=(area.min(), area.max()), hue=area, palette='coolwarm')

    plt.subplot(133)
    plt.title('alpha=1.0')
    sns.scatterplot(x, y, size=area, sizes=(area.min(), area.max()), hue=area, palette='coolwarm')

    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)

    return

# seaborn_01()

def seaborn_02():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            02
        Describe
            폰트설정, 기본설정
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "02. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    x_label = ['한국', '중국', '일본', '미국']

    y_2020 = [0.88, 1.3, 1.2, 1.4]
    plt.bar(x_label, y_2020, align='center', color='blue')
    plt.ylabel('명')
    plt.title('출산률')
    plt.show()

    sns.barplot(x_label, y_2020, alpha=0.8, palette='coolwarm')
    plt.ylabel('명')
    plt.title('출산률')
    plt.show()

    sns.barplot(y_2020, x_label, alpha=0.8, palette='coolwarm')
    plt.ylabel('명')
    plt.title('출산률')
    plt.show()

    print("\n", "=" * 3, "01.", "=" * 3)

    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/합계출산율_OECD.csv', delimiter=',', encoding='CP949')

    fig, axes = plt.subplots()

    print(data.info())


    print("\n", "=" * 3, "02.", "=" * 3)
    sns.barplot(data=data[1:2])

    # sns.barplot(data=data[0:2],
    #             y='1955',
    #             hue='국가별',  # 특정 컬럼값을 기준으로 나눠서 보고 싶을 때
    #             palette='pastel',  # pastel, husl, Set2, flare, Blues_d
    #             edgecolor=".6",  # edge 선명도 지정
    #             linewidth=2.5
    #             )
    # data.loc[data['증가'] > 50, '항목':'감소']
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)


seaborn_02()

# x_label = ['한국', '중국', '일본', '미국']
# x = np.arange(len(x_label))
#
# fig, axes = plt.subplots()
# y_2020 = [0.88, 1.3, 1.2, 1.4]
# y_2019 = [1.0, 1.5, 1.5, 1.6]
# width = 0.35
#
# axes.bar(x + width / 2, y_2020, width, align='center', alpha=0.9)
# axes.bar(x - width / 2, y_2019, width, align='center', alpha=0.9)
# plt.xticks(x)
# axes.legend(['2020', '2019'])
# axes.set_xticklabels(x_label)
# plt.show()


def seaborn_temp():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            temp
        Describe
            폰트설정, 기본설정
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "temp. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()
    # 데이터 설정
    df_00 = data_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# seaborn_temp()