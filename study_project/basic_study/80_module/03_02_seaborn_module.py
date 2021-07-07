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
            5. 데이터 프레임이 완성되있는 경우 활용하면 편하다.

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
    from matplotlib import font_manager, rc
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

    print("\n", "=" * 3, "01.", "=" * 3)

    titanic = sns.load_dataset('titanic')
    sns.barplot(data=titanic, x='sex', y='survived', hue='pclass', palette='spring')
    plt.show()


    print("\n", "=" * 3, "02. OECD 출산율 데이터", "=" * 3)

    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/합계출산율_OECD.csv', delimiter=',', encoding='CP949')
    plt.rcParams['figure.figsize'] = [15, 8]
    sns.barplot(data=data[1:2])
    plt.xlabel('한국 출산율')

    plt.show()

    print("\n", "=" * 3, "03. 세계 코로나 데이터", "=" * 3)

    plt.rcParams['figure.figsize'] = [10, 5]
    data2 = pd.read_csv(path + '/file_data/Covid-19-World-Dataset.csv', delimiter=',', encoding='CP949')
    sns.barplot(data=data2,  x='continent', y='new_cases')
    plt.show()

# seaborn_02()


def seaborn_03():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html



            area plot X 지원안함
            pie plot X  지원안함
        content
            03. line plot

        Describe
            line plot   : https://seaborn.pydata.org/generated/seaborn.lineplot.html
        sub Contents
            01. sin cos data
            02. 10 years of monthly airline passenger data
            03. 자기공명영상(Functional magnetic resonance imaging, fMRI)
    """
    print("\n", "=" * 5, "03. line plot. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01. sin cos data", "=" * 3)
    # grid style whitegrid, darkgrid, white, dark, ticks
    sns.set_style('whitegrid')

    x = np.arange(0, 10, 0.2)
    y_1 = 1 + np.sin(x)
    y_2 = 1 + np.cos(x)

    sns.lineplot(x, y_1, label='1+sin', color='blue')
    sns.lineplot(x, y_2, label='1+cos', color='red')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print("\n", "=" * 3, "02. 10 years of monthly airline passenger data", "=" * 3)

    fig, axes = plt.subplots(2, 2, sharey=True, tight_layout=True, figsize=(15, 6))
    flights = sns.load_dataset("flights")
    print(flights.head())


    may_flights = flights.query("month == 'May'")
    flights_wide = flights.pivot("year", "month", "passengers")
    flights_wide.head()

    axes[0, 0] = sns.lineplot(data=may_flights, x="year", y="passengers", ax=axes[0, 0])
    axes[0, 1] = sns.lineplot(data=flights_wide["May"], ax=axes[0, 1])
    axes[1, 0] = sns.lineplot(data=flights_wide, ax=axes[1, 0])
    axes[1, 1] = sns.lineplot(data=flights, x="year", y="passengers", ax=axes[1, 1])
    plt.show()

    # Assign a grouping semantic (hue, size, or style) to plot separate lines
    fig, axes = plt.subplots(2, 2, sharey=True, tight_layout=True, figsize=(15, 6))
    axes[0, 0] = sns.lineplot(ax=axes[0, 0], data=flights, x="year", y="passengers", hue="month")
    plt.show()


    print("\n", "=" * 3, "03. 자기공명영상(Functional magnetic resonance imaging, fMRI)", "=" * 3)
    # FMRI? : 자기공명영상(Functional magnetic resonance imaging, fMRI)는 혈류와 관련된 변화를 감지하여 뇌 활동을 측정하는 기술이다.
    # Each semantic variable can also represent a different column. For that, we’ll need a more complex dataset:
    fmri = sns.load_dataset("fmri")
    print(fmri.head())
    fig, axes = plt.subplots(2, 2, sharey=True, tight_layout=True, figsize=(15, 6))
    sns.lineplot(ax=axes[0, 0], data=fmri, x="timepoint", y="signal", hue="event")
    sns.lineplot(ax=axes[1, 0], data=fmri, x="timepoint", y="signal", hue="region", style="event")
    sns.lineplot(ax=axes[0, 1], data=fmri, x="timepoint", y="signal", hue="event", style="event", markers=True, dashes=False)
    sns.lineplot(ax=axes[1, 1], data=fmri, x="timepoint", y="signal", hue="event", err_style="bars", ci=68)
    plt.show()

# seaborn_03()


def seaborn_04():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            04. box plot
        Describe
            box plot    : https://seaborn.pydata.org/generated/seaborn.boxplot.html

        sub Contents
            01. 타이타닉 데이터
    """
    print("\n", "=" * 5, "04. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01. 타이타닉 데이터", "=" * 3)



    # 아웃라인, 마커설정
    outlier_marker = dict(markerfacecolor='r', marker='D')

    fig, axes = plt.subplots(2, 2, sharey=True, tight_layout=True, figsize=(15, 6), num='Figure Titanic')
    fig.suptitle('Figure Titanic', fontsize=15)
    titanic = sns.load_dataset('titanic')
    print(titanic.head())

    axes[0, 0].set_title('subplot 1')
    sns.boxplot(ax=axes[0, 0], data=titanic, x="pclass", y="age", hue="survived", flierprops=outlier_marker)
    plt.show()



    print("\n", "=" * 3, "02.", "=" * 3)
    tips = sns.load_dataset("tips")

    fig_box = plt.figure('1 graph')
    sns.boxplot(x=tips["total_bill"])
    plt.show()

    # sharey = y축 공유, sharex = x 축 공유
    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='Figure tips')
    fig.suptitle('tips data', fontsize=15)
    print(tips.head())

    axes[0, 0].set_title('subplot 1')
    axes[0, 1].set_title('subplot 2')
    axes[1, 0].set_title('subplot 3')
    axes[1, 1].set_title('subplot 4')

    # 기본 그래프
    sns.boxplot(ax=axes[0, 0], x="day",  y="total_bill", data=tips)

    # 범주 (hue)를 추가하여 2 개의 범주 형 변수별로 그룹화 하여 사용
    sns.boxplot(ax=axes[0, 1], x="day",  y="total_bill", hue="smoker", data=tips, palette="Set3")

    # 명시적인 순서를 전달하여 순서를 정하기
    sns.boxplot(ax=axes[1, 0], x="day",  y="total_bill", hue="time", data=tips, linewidth=2.5, order=['Sun', 'Sat', 'Fri', 'Thur'])
    sns.boxplot(ax=axes[1, 1], x="time", y="tip", data=tips, order=["Dinner", "Lunch"])
    plt.show()

    tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='weekend')
    fig.suptitle('tips weekend data', fontsize=15)
    print(tips.head())

    axes[0, 0].set_title('subplot 1')
    sns.boxplot(ax=axes[0, 0], x="day", y="total_bill", hue="weekend", data=tips, dodge=False)

    sns.boxplot(ax=axes[0, 1], x="day", y="total_bill", data=tips)
    # swarmplot : 상자 위에 데이터 포인트를 표시하는 데 사용 합니다
    sns.swarmplot(ax=axes[0, 1], x="day", y="total_bill", data=tips, color=".25")
    plt.show()

    # catplot  : FacetGrid에 범주 형 플롯을 그리기위한 그림 수준 인터페이스입니다.
    sns.catplot(x="sex", y="total_bill",
                hue="smoker", col="time",
                data=tips, kind="box",
                height=4, aspect=.7)
    plt.show()

    print("\n", "=" * 3, "03. iris Data", "=" * 3)
    iris = sns.load_dataset("iris")

    # sharey = y축 공유, sharex = x 축 공유
    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='iris data')
    fig.suptitle('tips data', fontsize=15)
    print(iris.head())
    sns.boxplot(ax=axes[0, 0], data=iris, orient="h", palette="Set2")

    plt.show()


# seaborn_04()


def seaborn_05():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            05. histogram
        Describe
            hist plot : https://seaborn.pydata.org/generated/seaborn.histplot.html
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "05. histogram. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    n = 100000
    bins = 50
    x = np.random.randn(n)
    # bins = 하나의 구간을 몇분할 할지
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # plt.subplots(nrows=2, ncols=2, sharex=True, , sharey=True)
    axes[0].hist(x, bins=50)
    axes[1].hist(x, bins=100)
    plt.show()

    # kde = 밀도, hist = 히스토그램 표현
    # sns.distplot(x, bins=50, kde=False, hist=True, color='r')
    #
    sns.distplot(x, bins=50, kde=True, hist=False, color='r')
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    penguins = sns.load_dataset("penguins")

    # sharey = y축 공유, sharex = x 축 공유
    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='penguins')
    fig.suptitle('penguins data', fontsize=15)
    print(penguins.head())

    axes[0, 0].set_title('subplot 1')
    axes[0, 1].set_title('subplot 2')
    axes[1, 0].set_title('subplot 3')
    axes[1, 1].set_title('subplot 4')

    sns.histplot(ax=axes[0, 0], data=penguins, x="flipper_length_mm")
    # binwidth 길이를 정해서 분할함
    sns.histplot(ax=axes[0, 1], data=penguins, x="flipper_length_mm", binwidth=3)
    # bins 사용할 총 빈 수를 정의함
    sns.histplot(ax=axes[1, 0], data=penguins, x="flipper_length_mm", bins=30)

    sns.histplot(ax=axes[1, 1], data=penguins, y="flipper_length_mm")
    plt.show()

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='penguins')
    fig.suptitle('penguins data', fontsize=15)

    # kde 커널 밀도 추정값을 추가여 분포 모양에 대한 보완 정보를 제공함
    axes[0, 0].set_title('kde : 커널 밀도 추정값')
    sns.histplot(ax=axes[0, 0], data=penguins, x="flipper_length_mm", kde=True)

    # x또는 둘 다 y할당 되지 않은 경우 데이터 세트는 와이드 형식으로 처리되고 각 숫자 열에 대해 히스토그램이 그려집니다.
    axes[0, 1].set_title('x, y 미할당시 와이드 형식으로 처리')
    sns.histplot(ax=axes[0, 1], data=penguins)

    # 색조 매핑을 사용하여 긴 형식의 데이터 세트에서 여러 히스토그램을 그릴 수 있다.
    axes[1, 0].set_title('색조 매핑으로 긴 형식의 데이터 세트에서 히스토그램 제작')
    sns.histplot(ax=axes[1, 0], data=penguins, x="flipper_length_mm", hue="species")
    # multiple="stack" 스택으로 쌓아서 올릴 수 있다.
    axes[1, 1].set_title('스택 형식으로 제작')
    sns.histplot(ax=axes[1, 1], data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
    plt.show()


    print("\n", "=" * 3, "03.", "=" * 3)

# seaborn_05()


def seaborn_06():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            06. titanic sample data
                titanic = sns.load_dataset("titanic")
        Describe
            survived    : 생존여부
            alive       : 생존여부 (영문)
            plcass      : 좌석등급
            class       : 좌석등급 (영문)
            sex         : 성별
            age         : 나이
            sibsp       : 형제자매 + 배우자 숫자
            parch       : 부모자식 숫자
            fare        : 요금
            who         : 사람구분
            deck        : 갑판?
            embark_town : 탑승 항구 (총 3군대)
            alone       : 혼자인지 여부

        sub Contents
            01. count plot
                숫자 기준 옵션
            02. distplot
                데이터의 밀도
    """
    print("\n", "=" * 5, "06. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()
    sns.set_style('darkgrid')
    titanic = sns.load_dataset("titanic")

    print("\n", "=" * 3, "01. count plot", "=" * 3)
    fig, axes = plt.subplots(2, 2, num='count plot', figsize=(15, 6), tight_layout=True)
    axes[0, 0].set_title('Graph Vertical')
    sns.countplot(ax=axes[0, 0], x='class', hue='who', data=titanic)

    axes[0, 1].set_title('Graph Horizontal')
    sns.countplot(ax=axes[0, 1], y='class', hue='who', data=titanic)

    #
    axes[1, 0].set_title('Graph Vertical')
    sns.countplot(ax=axes[1, 0], x='class', hue='who', data=titanic, palette='spring')

    axes[1, 1].set_title('Graph Horizontal')
    sns.countplot(ax=axes[1, 1], y='class', hue='who', data=titanic, palette='gist_ncar')
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# seaborn_06()


def seaborn_07():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            07. tips DataSet
                tips = sns.load_dataset("tips")
        Describe
            total_bill  : 총 합계 요금표
            tip         : 팁
            sex         : 성별
            smoker      : 흡연자 여부
            day         : 요일
            time        : 식사 시간
            size        : 식사 인원
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "07. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    sns.set_style('darkgrid')
    tips = sns.load_dataset("tips")

    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# seaborn_07()


def seaborn_08():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html
            palette doc : https://matplotlib.org/stable/tutorials/colors/colormaps.html

        content
            08. count plot :
        Describe
            항목별 개수를 세어주는 그래프다.
            알아서 해당 column을 구성하고 있는 value를 구분하여 보여준다.
            Doc : https://seaborn.pydata.org/generated/seaborn.countplot.html
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "08. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    sns.set_style('darkgrid')
    print("\n", "=" * 3, "01.", "=" * 3)
    titanic = sns.load_dataset("titanic")

    # fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='penguins')

    fig, axes = plt.subplots(2, 2, num='count plot', figsize=(15, 6) , tight_layout=True)
    axes[0, 0].set_title('Graph Vertical')
    sns.countplot(ax=axes[0, 0], x='class', hue='who', data=titanic)

    axes[0, 1].set_title('Graph Horizontal')
    sns.countplot(ax=axes[0, 1], y='class', hue='who', data=titanic)

    #
    axes[1, 0].set_title('Graph Vertical')
    sns.countplot(ax=axes[1, 0], x='class', hue='who', data=titanic, palette='spring')

    axes[1, 1].set_title('Graph Horizontal')
    sns.countplot(ax=axes[1, 1], y='class', hue='who', data=titanic, palette='gist_ncar')

    plt.show()
    print("\n", "=" * 3, "02.", "=" * 3)

    fig, axes = plt.subplots(2, 2, num='count plot', figsize=(15, 6), tight_layout=True)
    sns.countplot(ax=axes[0, 0], x='sex', data=titanic, palette='gist_ncar')
    sns.countplot(ax=axes[0, 1], x='sex', hue='pclass', data=titanic, palette='gist_ncar')

    sns.countplot(ax=axes[1, 0], x='embark_town', hue='pclass', data=titanic, palette='gist_ncar')
    sns.countplot(ax=axes[1, 1], x='embark_town', hue='alive', data=titanic, palette='gist_ncar')

    plt.show()
    print("\n", "=" * 3, "03.", "=" * 3)


# seaborn_08()

def seaborn_09():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            09. distplot
                데이터의 분포, 밀도를 확인 그래프
                rugplot
        Describe
            matplotlib의 hist, kdeplot을 통합한 그래프이다.
            데이터의 분포, 밀도를 확인할 수 있다.
            Doc : https://seaborn.pydata.org/generated/seaborn.displot.html
        sub Contents
            01. distplot : rug, hist , vertical, color
            02.
    """
    print("\n", "=" * 5, "09. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()
    # 데이터 설정


    print("\n", "=" * 3, "01.", "=" * 3)
    fig, axes = plt.subplots(2, 2, num='dist_plot', sharey=False, tight_layout=True, figsize=(12, 8))

    array_data = np.random.randn(100)
    axes[0, 0].set_title('array data graph')
    sns.distplot(ax=axes[0, 0], x=array_data)

    series_data = pd.Series(array_data, name='x var')
    axes[0, 1].set_title('series data graph')
    sns.distplot(ax=axes[0, 1], x=series_data)

    axes[1, 0].set_title('rug data graph')
    sns.distplot(ax=axes[1, 0], x=array_data, rug=True, hist=False)

    axes[1, 1].set_title('kde : Kernel density & vertical=True')
    sns.distplot(ax=axes[1, 1], x=array_data, kde=True, rug=True, hist=False, vertical=True, color='y')

    plt.show()

# seaborn_09()


def seaborn_10():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            10. pairplot
            doc : https://seaborn.pydata.org/generated/seaborn.pairplot.html
        Describe
            pariplot은 그리드(grid) 형태로 각 집합의 조합에 대해 히스토그램과 분포도를 그린다.
            숫자형 Column에 대해서만 가능

        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "10. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    tips = sns.load_dataset("tips")
    sns.pairplot(tips)
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    sns.pairplot(tips, hue='size')
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)
    sns.pairplot(tips, hue='size', palette='spring', height=3)
    plt.show()


# seaborn_10()


def seaborn_11():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            11. violinplot
            doc : https://seaborn.pydata.org/generated/seaborn.violinplot.html
        Describe
            바이올린처럼 생긴 차트
            column에 대한 데이터의 비교 분포도를 확인할 수 있다.
                - 곡선진 부분은 데이터의 분포를 나타낸다.
                - 양쪽 끝 뾰족한 부분은 데이터의 최소값과 최대값을 나타낸다.
        sub Contents
            01. 기본 그리기
    """
    print("\n", "=" * 5, "11. 기본 그리기. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    tips = sns.load_dataset('tips')
    sns.violinplot(x=tips['total_bill'])
    plt.show()

    print("\n", "=" * 3, "02. 비교 분포 ", "=" * 3)
    fig, axes = plt.subplots(2, 2, num='violin_plot', figsize=(12, 8))

    axes[0, 0].set_title('violin_plot_vertical')
    axes[0, 1].set_title('violin_plot_horizontal')
    sns.violinplot(ax=axes[0, 0], x='day', y='total_bill', data=tips)
    sns.violinplot(ax=axes[0, 1], y='day', x='total_bill', data=tips)


    axes[1, 0].set_title('hue option')
    axes[1, 1].set_title('split opption')

    sns.violinplot(ax=axes[1, 0], x='day', y='total_bill', hue='smoker', data=tips)
    sns.violinplot(ax=axes[1, 1], x='day', y='total_bill', hue='smoker', split=True, data=tips)
    plt.show()
    print("\n", "=" * 3, "03.", "=" * 3)


# seaborn_11()


def seaborn_12():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            12. lmplot
        Describe
            doc : https://seaborn.pydata.org/generated/seaborn.lmplot.html
            column 간의 선형관계를 확인하기 용이한 차트
            outlier도 체크가 가능함.

        sub Contents
            01. 기본 그리기
    """
    print("\n", "=" * 5, "12. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    tips = sns.load_dataset('tips')
    sns.lmplot(x='total_bill', y='tip', height=6, data=tips)
    plt.show()


    print("\n", "=" * 3, "02.", "=" * 3)
    sns.lmplot(x='total_bill', y='tip', hue='smoker', height=6, data=tips)
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)
    # col='컬럼 기준', col_wrap 한줄에 표기할 컬럼이 갯수
    sns.lmplot(x='total_bill', y='tip', hue='smoker', col='day', col_wrap=2, height=6, data=tips)
    plt.show()


# seaborn_12()


def seaborn_13():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            13. relplot
        Describe
            doc : https://seaborn.pydata.org/generated/seaborn.relplot.html
            두 column간 상관관계를 볼 수 있다. 선형관계는 그려주지 않는다.
            선형관계가 보고싶으면 lmplot 그냥 간단히 보고싶으면 relplot

        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "13. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()
    tips = sns.load_dataset('tips')

    print("\n", "=" * 3, "01.", "=" * 3)
    sns.relplot(x='total_bill', y='tip', hue='day', data=tips)
    plt.show()
    print("\n", "=" * 3, "02.", "=" * 3)
    sns.relplot(x='total_bill', y='tip', hue='day', col='time', col_wrap=2, data=tips)
    plt.show()

    sns.relplot(x='total_bill', y='tip', hue='day', row='sex', col='time',  data=tips, height=5)
    plt.show()
    print("\n", "=" * 3, "03.", "=" * 3)


# seaborn_13()


def seaborn_14():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            14. jointplot
            scatter(산점도)와 histogram(분포)을 동시에 그려줍니다.
            숫자형 데이터만 표현 가능

        Describe
            doc : https://seaborn.pydata.org/generated/seaborn.jointplot.html
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "14. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    tips = sns.load_dataset('tips')
    sns.jointplot(x='total_bill', y='tip', height=5, data=tips)
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    # 선형 표현 그래프
    sns.jointplot(x='total_bill', y='tip', height=5, data=tips, kind='reg')
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)
    # 핵스 그래프
    sns.jointplot(x='total_bill', y='tip', height=5, data=tips, kind='hex')
    plt.show()

    # 등고선 그래프
    sns.jointplot(x='total_bill', y='tip', height=5, data=tips, kind='kde')
    plt.show()


seaborn_14()


def seaborn_heatmap():
    """
        subject
            Data analysis module - visualization library
        topic
            seaborn 모듈
            seaborn doc : https://seaborn.pydata.org/
            pyplot doc  : https://matplotlib.org/2.0.2/api/pyplot_api.html

        content
            heatmap
        Describe
            색상으로 표현할 수 있는 다양한 정보를 일정한 이미지위에 열분포 형태의 비쥬얼한 그래픽으로 출력력
                1. pivot x, y 축 으로 만든후 시각화 할때
                2. 데이터 칼럼간의 상관관계를 보고 싶을때
       sub Contents
            01. 기본 heatmap
    """
    print("\n", "=" * 5, "1301. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    rand_data = np.random.rand(10, 12)
    print(rand_data)

    sns.heatmap(rand_data, annot=True)
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    # fig, axes = plt.subplots(2, 2, num='dist_plot', sharey=False, tight_layout=True, figsize=(12, 8))
    tips = sns.load_dataset('tips')
    pivot = tips.pivot_table(index='day', columns='size', values='tip')
    print(pivot)
    sns.heatmap(pivot, annot=True)
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)
    titanic = sns.load_dataset('titanic')
    sns.heatmap(titanic.corr(), annot=True, cmap='spring')
    plt.show()

# seaborn_temp01()

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