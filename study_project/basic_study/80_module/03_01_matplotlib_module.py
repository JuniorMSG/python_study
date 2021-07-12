"""
    subject
        Data analysis module - visualization library
    topic
        matplotlib 모듈
    Describe
        파이썬 기반 시각화 (visualization) 라이브러리
        matplotlib.pyplot
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        pandas도 matplotlib를 내장하고 있다.

        장점
            1. 파이썬 표준 시각화 도구라고 불릴만큼 다양한 기능 지원
            2. 세부 옵션을 통하여 스타일링 가능
            3. 다양한 그래프 그릴 수 있음
            4. pandas와 연동이 쉬움

        단점
            1. 한글에 대한 완벽한 지원 X
            2. 세부 기능이 많은 대신 설정해야 할 값이 많다.
        https://matplotlib.org/
        https://matplotlib.org/stable/contents.html

    Contents
        font_set : 폰트 설정 함수
        data_set : 데이터 설정 함수
        01. 기본설정
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import numpy as np
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

def matplotlib_01():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
        content
            01. 기본설정
        Describe
            폰트설정, 기본설정
        sub Contents
            01. 폰트설정
            02. plt.show()  = 출력
            03. 사이즈 크기 조절
    """
    print("\n", "=" * 5, "01. 기본설정. ", "=" * 5)

    print("\n", "=" * 3, "01.font 설정 : 한글 깨짐 방지", "=" * 3)
    font_path = "C:\Windows\Fonts\H2HDRM.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font)


    print("\n", "=" * 3, "02. 출력", "=" * 3)

    # 데이터 설정
    df_00 = data_set()
    df_00.groupby('연도').count().plot(kind='pie', y='분양가격', title='pie plot')


    print("\n", "=" * 3, "03. 사이즈 크기 조절", "=" * 3)
    plt.rcParams["figure.figsize"] = (10, 5)
    df_00.groupby('연도').count().plot(kind='pie', y='분양가격', title='pie plot')
    plt.show()

# matplotlib_01()


def matplotlib_02():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            02. 그래프 그리기
        Describe
            단일 , 다중 그래프
        sub Contents
            01. 단일 그래프
            02. 다중 그래프 (multiple graphs)
            03. 다중 그래프 서브 plot
    """
    print("\n", "=" * 5, "01. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()


    print("\n", "=" * 3, "01. 단일 그래프", "=" * 3)

    data = np.arange(1, 150)
    plt.plot(data)
    plt.show()

    print("\n", "=" * 3, "02. 다중 그래프", "=" * 3)

    data_02_01 = np.arange(1, 150)
    data_02_02 = data_02_01 * 10

    plt.plot(data_02_01)

    # 새로운 캔버스 생성
    plt.figure()
    plt.plot(data_02_02)
    plt.show()

    print("\n", "=" * 3, "03. 다중 그래프 서브 plot", "=" * 3)
    data_03_01 = np.arange(1, 150)

    # plt.subplot(row, column, index)
    plt.subplot(221)
    plt.plot(data_03_01)

    plt.subplot(222)
    plt.plot(data_03_01 + 50)

    plt.subplot(223)
    plt.plot(data_03_01 + 150)

    plt.subplot(224)
    plt.plot(data_03_01 * 10)

    # plt.subplots
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(data_03_01)
    axes[0, 1].plot(data_03_01 // 15)
    axes[1, 0].plot(data_03_01 ** 4)
    axes[1, 1].plot(data_03_01 % 10)

    plt.tight_layout()
    plt.show()

# matplotlib_02()


def matplotlib_03():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            03. 주요 스타일 옵션
        Describe
            스타일 옵션
        sub Contents
            01. 타이틀 설정 : plt.title('')
                option : fontsize
            02. X, Y 축 Label, Tick(rotation) 설정
                라벨 설정         : plt.xlabel, plt.ylabel
                틱(돌리기)        : plt.xticks, plt.yticks
                범례(Legend)     : plt.legent(
                한계점설정(limit) : plt.xlim, plt.ylim
            03. plot 스타일 세부 설정
                https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
                plt.plot(data, marker='o', markersize=5, linestyle='-', color='b', alpha=0.5)

                marker='o', markersize=5, linestyle='-', color='b', alpha=0.2
                marker, markersize  : 마커 스타일, 마커 사이즈
                linestyle           : 라인 스타일
                color               : 컬러
                alpha               : 투명도
                plt.grid()          : 격자 설정
    """
    print("\n", "=" * 5, "01. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()
    # 데이터 설정


    print("\n", "=" * 3, "01.", "=" * 3)
    data = np.arange(1, 10)
    plt.title('주요 스타일 옵션', fontsize=15)
    plt.plot(data)
    plt.show()
    print("\n", "=" * 3, "02.", "=" * 3)

    plt.title('x축 y축 라벨 설정', fontsize=15)

    plt.plot(data * 2)
    plt.plot(data ** 2)
    plt.plot(np.log(data))
    #각 축에 라벨 설정 옵션
    plt.xlabel('x축')
    plt.ylabel('y축')

    # 각도를 설정해주는 옵션임 (하나의 틱이 길때 사용하면 됨.)
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)

    # 범례 설정
    plt.legend(['10 * 2', '10 ** 2', 'log'], fontsize=15)

    # limit 설정
    plt.xlim(0, 10)
    plt.ylim(0, 200)

    plt.show()
    print("\n", "=" * 3, "03. 세부 설정", "=" * 3)

    plt.title('세부 설정', fontsize=15)

    plt.plot(data * 2, marker='o', markersize=5, linestyle='-', color='b', alpha=0.2)
    plt.plot(data * 3, marker='o', markersize=5, linestyle='-', color='b', alpha=0.5)
    plt.plot(data * 4, marker='o', markersize=5, linestyle='-', color='b', alpha=0.8)
    plt.plot(data ** 2, marker='v', markersize=10, linestyle='-.', color='r', alpha=0.5)
    plt.plot(np.log(data), marker='+', markersize=15, linestyle=':', color='c', alpha=0.7)
    #각 축에 라벨 설정 옵션
    plt.xlabel('x축')
    plt.ylabel('y축')

    # 각도를 설정해주는 옵션임 (하나의 틱이 길때 사용하면 됨.)
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)

    # 범례 설정
    plt.legend(['10 * 2', '10 ** 2', 'log'], fontsize=15)
    plt.grid()

    # 그래프 이미지로 저장
    directory = 'image/matplotlib_03/'
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    createFolder(directory)
    plt.savefig(directory + 'graph.png', dpi=300)
    plt.show()


# matplotlib_03()

def matplotlib_04():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            04. 산점도 그래프
        Describe
            산점도(散點圖,scatter plot, scatterplot, scatter graph, scatter chart, scattergram, scatter diagram)는
            직교 좌표계(도표)를 이용해 좌표상의 점(點)들을 표시함으로써 두 개 변수 간의 관계를 나타내는 그래프 방법이다.
            도표 위에 두 변수 엑스(X)와 와이(Y) 값이 만나는 지점을 표시한 그림. 이 그림을 통해 두 변수 사이의 관계를 알 수 있다.
            matplotlib.pyplot.scatter
            (x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
            linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs)[source]

        sub Contents
            01. scatter (산점도 그래프)
            02. scatter - cmap
    """
    print("\n", "=" * 5, "01. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    # 데이터 생성
    data_ran = np.random.rand(100)
    data_arange = np.arange(50)
    print(data_ran)
    print(data_arange)


    print("\n", "=" * 3, "01. scatter (산점도 그래프)", "=" * 3)
    x = np.random.rand(100)
    y = np.random.rand(100)
    colors = np.arange(100) / 10
    area = x * y * 1000

    plt.scatter(x, y, s=area, c=colors)
    plt.show()
    print("\n", "=" * 3, "02. scatter - cmap", "=" * 3)

    arr = np.random.standard_normal((8, 50))
    print(arr)

    plt.subplot(2, 2, 1)
    plt.scatter(arr[0], arr[1], c=arr[1], cmap='spring')
    plt.title('spring')
    # plt.spring()

    plt.subplot(2, 2, 2)
    plt.scatter(arr[2], arr[3], c=arr[3], cmap='summer')
    plt.title('summer')
    # plt.summer()

    plt.subplot(2, 2, 3)
    plt.scatter(arr[4], arr[5], c=arr[5], cmap='autumn')
    plt.title('autumn')
    # plt.autumn()

    plt.subplot(2, 2, 4)
    plt.scatter(arr[6], arr[7], c=arr[7], cmap='winter')
    plt.title('winter')
    # plt.winter()

    plt.tight_layout()
    plt.show()


matplotlib_04()


def matplotlib_05():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            05. Barplot, Barhplot
        Describe
            바 그래프
        sub Contents
            01. 바형태 그래프 그리기
                plt.bar(x, y)
            02. horizontal 바형태 그래프 그리기
                plth.bar(x, y)
            03. 비교 그래프 그리기

    """
    print("\n", "=" * 5, "01. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01. 바형태 그래프 그리기", "=" * 3)
    x_label = ['한국', '중국', '일본', '미국']


    y_2020 = [0.88, 1.3, 1.2, 1.4]
    plt.bar(x_label, y_2020, align='center', color='blue')
    plt.xticks(x_label)
    plt.ylabel('명')
    plt.title('출산률')
    plt.show()

    print("\n", "=" * 3, "02. horizontal 바형태 그래프 그리기", "=" * 3)

    plt.barh(x_label, y_2020, align='center', color='blue')
    plt.xticks(y_2020)
    plt.ylabel('명')
    plt.title('출산률')
    plt.show()

    print("\n", "=" * 3, "03. 비교 그래프 그리기", "=" * 3)
    x_label = ['한국', '중국', '일본', '미국']
    x = np.arange(len(x_label))

    fig, axes = plt.subplots()
    y_2020 = [0.88, 1.3, 1.2, 1.4]
    y_2019 = [1.0, 1.5, 1.5, 1.6]
    width = 0.35

    axes.bar(x + width/2, y_2020, width, align='center', alpha=0.9)
    axes.bar(x - width/2, y_2019, width, align='center', alpha=0.9)
    plt.xticks(x)
    axes.legend(['2020','2019'])
    axes.set_xticklabels(x_label)
    plt.show()


# matplotlib_05()


def matplotlib_06():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            06. lineplot
        Describe
            선 그래프
        sub Contents
            01. 1개 그래프 그리기
            02. multi 그래프 그리기
            03. 기타 옵션 넣어서 그리기
    """
    print("\n", "=" * 5, "06. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    x = np.arange(0, 10, 0.1)
    y = np.cos(x)

    plt.plot(x, y)
    plt.xlabel('x value', fontsize=15)
    plt.ylabel('y value', fontsize=15)
    plt.title('cos graph', fontsize=18)
    plt.grid()
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)

    x = np.arange(0, 10, 0.1)
    y_1 = np.cos(x)
    y_2 = np.sin(x)
    # y_3 = 1 - np.tan(x)

    plt.plot(x, y_1, label='cos')
    plt.plot(x, y_2, label='sin')
    # plt.plot(x, y_3, label='tan')
    plt.xlabel('x value', fontsize=15)
    plt.ylabel('y value', fontsize=15)
    plt.title('sin cos graph', fontsize=18)
    plt.legend()
    plt.grid()
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)

    x = np.arange(0, 10, 0.1)
    y_1 = np.cos(x)
    y_2 = np.sin(x)
    # y_3 = 1 - np.tan(x)

    plt.plot(x, y_1, label='cos', marker='D', linestyle=':')
    plt.plot(x, y_2, label='sin', marker='<', linestyle='-.')
    # plt.plot(x, y_3, label='tan')
    plt.xlabel('x value', fontsize=15)
    plt.ylabel('y value', fontsize=15)
    plt.title('sin cos graph', fontsize=18)
    plt.legend()
    plt.grid()
    plt.show()

# matplotlib_06()


def matplotlib_07():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            07. Areaplot

        Describe
            에어리가 색칠된 그래프
            겹치는 부분을 나타낼때 좋다.
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "07. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()
    # 데이터 설정
    x = np.arange(1, 31)
    y_1 = np.random.randint(low=4, high=30, size=30)
    y_2 = np.random.randint(low=4, high=20, size=30)
    y_3 = np.random.randint(low=4, high=10, size=30)

    print("\n", "=" * 3, "01.", "=" * 3)
    plt.fill_between(x, y_1, color="red", alpha=0.3)
    plt.plot(x, y_1, color="red", alpha=0.8)
    plt.show()


    print("\n", "=" * 3, "02.", "=" * 3)
    plt.fill_between(x, y_1, color="red", alpha=0.2)
    plt.fill_between(x, y_2, color="green", alpha=0.2)
    plt.fill_between(x, y_3, color="blue", alpha=0.2)
    plt.show()
    print("\n", "=" * 3, "03.", "=" * 3)

matplotlib_07()


def matplotlib_08():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            08. Histogram
        Describe
            데이터의 밀도를 보고 싶을때 사용한다.

        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "08. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()
    n = 100000
    bins = 30


    print("\n", "=" * 3, "01.", "=" * 3)
    # 균일 분포의 정수 난수 1개 생성
    x = np.random.randint(n)
    print(x)
    plt.hist(x, bins=bins)
    plt.show()

    # 0부터 1사이의 균일 분포에서 난수 array생성
    x = np.random.rand(n)
    plt.hist(x, bins=bins)
    plt.show()

    # 가우시안 표준 정규 분포에서 난수 array생성
    x = np.random.randn(n)
    plt.hist(x, bins=bins)
    plt.show()

    print("\n", "=" * 3, "02. 다중 Histogram", "=" * 3)
    x = np.random.randn(n)
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    axs[0].hist(x, bins=bins)
    axs[1].hist(x, bins=bins*2)
    axs[2].hist(x, bins=bins*5)
    plt.show()
    print("\n", "=" * 3, "03. Density 표기 (%)", "=" * 3)

    x = np.random.randn(n)
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    axs[0].hist(x, bins=bins, density=True)
    axs[1].hist(x, bins=bins, density=True, cumulative=True)
    plt.show()

# matplotlib_08()


def matplotlib_09():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            09. Pie Char
        Describe
            점유율을 표현하고 싶을때 많이 사용
            explode     : 파이에서 툭 튀어저 나온 비율
            autopct     : 퍼센트 자동으로 표기
            shadow      : 그림자 표시
            startangle  : 파이를 그리기 시작할 각도
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "09. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    data_labels = ["SBUX", 'KO', 'PEP']
    sizes = [20, 40.5, 39.5]
    explode = [0.3, 0, 0]

    patches, texts, autoexts = plt.pie(sizes, explode=explode
                                       , labels=data_labels
                                       , autopct='%1.1f%%'
                                       , shadow=True
                                       , startangle=90
                                       )
    plt.title('beverage chart')
    for t in texts:
        t.set_fontsize(12)
        t.set_color('gray')

    for t in autoexts:
        t.set_fontsize(18)
        t.set_color('white')

    plt.show()
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


# matplotlib_09()

def matplotlib_10():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html
        content
            10. Boxplot
        Describe
            폰트설정, 기본설정
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "10. 기본설정. ", "=" * 5)

    # 폰트 설정
    font_set()

    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    fl_high = np.random.rand(20) * 100 + 100
    fl_low  = np.random.rand(20) * -100
    data = np.concatenate((spread, center, fl_high, fl_low))
    # concatenate - 순차적으로 엮다 - 배열 합치기 역활을 수행한다.


    print("\n", "=" * 3, "01.", "=" * 3)
    data_des = pd.DataFrame(data).describe()[0]
    IQR = (data_des['75%'] - data_des['25%']) * 1.5
    print("3Q", data_des['75%'])
    print("1Q", data_des['25%'])
    print("Median", data_des['50%'])
    print("IQR", IQR)
    print("Lower Whisker :", data_des['25%'] - IQR)
    print("Upper Whisker :", data_des['75%'] + IQR)
    print("possible outliers를 보기 위해서 적절")

    plt.boxplot(data)
    plt.tight_layout()
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)

    spread = np.random.rand(50) * 100
    center = np.ones(25) * 40
    fl_high = np.random.rand(20) * 100 + 100
    fl_low  = np.random.rand(20) * -100
    data2 = np.concatenate((spread, center, fl_high, fl_low))

    d2 = [data, data2, data[::5]]

    plt.boxplot(d2)
    plt.tight_layout()
    plt.show()


    print("\n", "=" * 3, "03.", "=" * 3)
    marker = dict(markerfacecolor='r', marker='D')
    plt.boxplot(d2,
                labels=['data1', 'data2', 'data2-2'],
                vert=False, flierprops=marker)
    plt.tight_layout()
    plt.show()


# matplotlib_10()

def matplotlib_11():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            11. 3D 그래프
        Describe
            폰트설정, 기본설정
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "11. 기본설정. ", "=" * 5)
    from mpl_toolkits import mplot3d
    # 폰트 설정
    font_set()
    fig = plt.figure()
    ax = plt.axes(projection="3d")




    print("\n", "=" * 3, "01.", "=" * 3)
    z = np.linspace(0, 15, 1000)
    x = np.sin(z)
    y = np.cos(z)
    ax.plot(x, y, z, 'gray')
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    sample_size = 200
    x = np.cumsum(np.random.normal(0, 1, sample_size))
    y = np.cumsum(np.random.normal(0, 1, sample_size))
    z = np.cumsum(np.random.normal(0, 1, sample_size))

    ax.plot3D(x, y, z, alpha=0.8, marker="D")
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    sample_size = 1000
    x = np.cumsum(np.random.normal(0, 10, sample_size))
    y = np.cumsum(np.random.normal(0, 10, sample_size))
    z = np.cumsum(np.random.normal(0, 10, sample_size))

    ax.scatter(x, y, z, c=z, s=20, alpha=0.5, cmap='spring', marker="D")
    plt.show()

    print("\n", "=" * 3, "04. 등고선", "=" * 3)
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    x, y = np.meshgrid(x, y)
    z = np.sin(np.sqrt(x**2 + y**2))

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection="3d")

    ax.contour3D(x, y, z, 20, cmap='spring')
    plt.show()


# matplotlib_11()


def matplotlib_12():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
        content
            12. imshow
        Describe
            이미지 데이터와 유사하게 행과 열을 가진 2차원의 데이터를 시각화 할 때 사용한다.
            load_digits : 0 ~ 16 값을 가지는 array로 이루어져 있다.
            1개의 array는 8*8 배열 안에 표현되어 있으며 숫자는 0~9까지 이루어져 있다.
        sub Contents
            01. 폰트설정
    """
    print("\n", "=" * 5, "12. 기본설정. ", "=" * 5)
    # pip install scikit-learn
    from sklearn.datasets import load_digits

    # 폰트 설정
    font_set()
    digits = load_digits()
    data_img = digits.images[:10]

    # 1채널
    print(data_img[0])


    print("\n", "=" * 3, "01.", "=" * 3)

    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, figsize=(12, 6), sharey=True)
    for i in range(10):
        axes[i//5][i%5].imshow(data_img[i], cmap="spring")
        axes[i//5][i%5].set_title(str(i), fontsize=20)

    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


# matplotlib_12()

def matplotlib_temp():
    """
        subject
            Data analysis module - visualization library
        topic
            matplotlib 모듈
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#module-matplotlib.pyplot
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

# matplotlib_temp()