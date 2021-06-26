"""
    subject
        Data analysis module
    topic
        Pandas 모듈 시각화 (visualization)

    Describe
        시각화는 숫자, 문자형으로 된 데이터로부터
        이해하기 쉽도록 직관적인 시각적 정보를 만들어 내는것

        EDA (Exploratory Data Analysis)
        수집한 데이터가 들어왔을 때, 이를 다양한 각도에서 관찰하고 이해하는 과정.


    Contents
        01. 시각화
        02. 파일 읽기
        03. Ctrl + D를 누르면 하나씩 선택하여 수정이 가능함
        046 데이터 선택
"""


def pandas_vis_default_setting():
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import pandas as pd
    import numpy as np
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/주택도시보증공사_전국 신규 민간아파트 분양가격 동향_20200430.csv', delimiter=',', encoding='CP949')

    font_path = '/font/NanumBarunGothic.tif'
    font = fm.FontProperties(fname=font_path, size=0)

    print("Matplotlib version", matplotlib.__version__)

    fig = plt.figure()
    fig.suptitle('figure sample plots')

    fig, ax_lst = plt.subplots(2, 2, figsize=(8, 5))
    fig.set_size_inches(10, 10)


    ax_lst[0][0].plot([1, 2, 3, 4], 'ro-')
    ax_lst[0][1].plot(np.random.randn(4, 10), np.random.randn(4, 10), 'bo--')
    ax_lst[1][0].plot(np.linspace(0.0, 5.0), np.cos(2 * np.pi * np.linspace(0.0, 5.0)))
    ax_lst[1][1].plot([3, 5], [3, 5], 'bo:')

    plt.show()

    return

# pandas_vis_default_setting()


def pandas_vis_default_setting():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            pandas_vis_default_setting. select_dtypes
        Describe
            데이터 타입별 선택하는 메소드
        sub Contents
            01. font 설정
    """
    print("\n", "=" * 5, "pandas_vis_default_setting. ", "=" * 5)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc
    import pandas as pd
    import numpy as np
    import os

    # 데이터 가져오기
    path = os.path.dirname(os.path.abspath(__file__))

    print("\n", "=" * 3, "01. font 설정", "=" * 3)

    # 설정 해야 한글 안깨짐
    font_path = "C:\Windows\Fonts\H2HDRM.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

    print("\n", "=" * 3, "02. 기본 형태", "=" * 3)
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.suptitle('plt plot')

    df_00 = pd.DataFrame([1, 2, 3], [4, 5, 6])
    df_00.plot(kind='bar', title='bar Graph')
    df_00.plot(kind='barh', title='barh Graph')
    df_00.plot(kind='line', title='line Graph')
    df_00.plot(kind='area', title='area Graph')
    df_00.plot(kind='hist', title='hist Graph')
    df_00.plot(kind='kde', title='kde Graph')

    plt.show()

# pandas_vis_default_setting()


def pandas_vis_01():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            pandas_vis_01
        Describe
            데이터 타입별 선택하는 메소드
        sub Contents
            01. 히스토그램
            02. 커널 밀도 그래프
            03. Hexbin
    """
    print("\n", "=" * 5, "pandas_vis_kinds_01. ", "=" * 5)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc
    import pandas as pd
    import numpy as np
    import os

    font_path = "C:\Windows\Fonts\H2HDRM.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))

    print("\n", "=" * 3, "03.히스토그램", "=" * 3)
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

    print("\n", "=" * 3, "01. 히스토그램은 분포, 빈도를 시각화하여 보여준다.", "=" * 3)

    # 히스토그램은 분포, 빈도를 시각화하여 보여준다.
    df_01.plot(kind='hist', y='분양가격', title='히스토그램')

    print("\n", "=" * 3, "02. 커널 밀도 그래프", "=" * 3)
    # 히스토그램과 유사하게 밀도를 보여주는 그래프
    # 히스토그램과 유사한 모양새를 갖추고 있으며 부드러운 라인을 가지고 있다.
    # No module named 'scipy' => pip install scipy
    df_01.plot(kind='kde', y="분양가격", title='커널 밀도 그래프')


    print("\n", "=" * 3, "03. Hexbin", "=" * 3)
    # 고밀도 산점도 그래프
    # x와 y키 값을 넣어 주어야 한다. 둘다 numeric type
    # 데이터의 밀도를 추정한다.
    df_01.plot(kind='hexbin', x='분양가격', y='연도', title='Hexbin 그래프', gridsize=20)
    plt.show()


# pandas_vis_01()

def pandas_vis_02():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            boxplot
        Describe
            데이터 타입별 선택하는 메소드
        sub Contents
            01.
    """
    print("\n", "=" * 5, "pandas_vis_02. boxplot ", "=" * 5)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc
    import pandas as pd
    import numpy as np
    import os

    print("\n", "=" * 3, "02. boxplot", "=" * 3)
    font_path = "C:\Windows\Fonts\H2HDRM.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

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


    print("\n", "=" * 3, "01.", "=" * 3)
    print(df_01['분양가격'].describe())
    # IQR = (3Q - 1Q) * 1.5
    IQR = (df_01['분양가격'].describe()['75%'] - df_01['분양가격'].describe()['25%'])*1.5
    print('IQR = ', IQR)
    box_max = IQR + df_01['분양가격'].describe()['75%']
    box_min = df_01['분양가격'].describe()['25%'] - IQR

    print('boxplot Max Value ',  box_max)
    print('boxplot Min Value ',  box_min)
    df_01.plot(kind='box', y='분양가격')

    print("\n", "=" * 3, "02.", "=" * 3)

    df_02 = df_01.copy()
    df_02_seoul = df_02.loc[df_02['지역'] == '서울']
    df_02_seoul.plot(kind='box', y='분양가격')

    IQR = (df_02_seoul['분양가격'].describe()['75%'] - df_02_seoul['분양가격'].describe()['25%'])*1.5
    print('IQR = ', IQR)
    box_max = IQR + df_02_seoul['분양가격'].describe()['75%']
    box_min = df_02_seoul['분양가격'].describe()['25%'] - IQR
    print('boxplot Max Value ',  box_max)
    print('boxplot Min Value ',  box_min)

    plt.show()
    print("\n", "=" * 3, "03.", "=" * 3)


# pandas_vis_02()


def pandas_vis_03():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            boxplot
        Describe
            데이터 타입별 선택하는 메소드
        sub Contents
            01.
    """
    print("\n", "=" * 5, "pandas_vis_03. boxplot ", "=" * 5)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc
    import pandas as pd
    import numpy as np
    import os

    print("\n", "=" * 3, "02. boxplot", "=" * 3)
    font_path = "C:\Windows\Fonts\H2HDRM.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

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

    df_02 = df_01.copy()
    df_02_seoul = df_02.loc[df_02['지역'] == '서울']
    df_02_seoul.plot(kind='box', y='분양가격')


    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


    plt.show()


# pandas_vis_03()



def pandas_vis_04():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            vis_04. 다양한 그래프 모양 1
        Describe
            데이터 타입별 선택하는 메소드
        sub Contents
            01. area graph
            02. pie plot        (파이 그래프)
            03. scatter plot    (산점도 그래프)
    """

    print("\n", "=" * 5, "vis_04. ", "=" * 5)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc
    import pandas as pd
    import numpy as np
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))

    font_path = "C:\Windows\Fonts\H2HDRM.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

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

    df_02 = df_01.copy()

    print("\n", "=" * 3, "01. area graph ", "=" * 3)
    df_02.groupby('월').count().plot(kind='line', y='분양가격', title='line graph')
    df_02.groupby('월').count().plot(kind='area', y='분양가격', title='area graph')

    print("\n", "=" * 3, "02. pie plot", "=" * 3)
    df_02.groupby('연도').count().plot(kind='pie', y='분양가격', title='pie plot')

    print("\n", "=" * 3, "03. scatter plot", "=" * 3)
    df_02.plot(kind='scatter', x='월', y='분양가격', title='scatter plot')
    plt.show()

pandas_vis_04()


def pandas_temp():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            temp. select_dtypes
        Describe
            데이터 타입별 선택하는 메소드
        sub Contents
            01.
    """
    print("\n", "=" * 5, "temp. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/주택도시보증공사_전국 신규 민간아파트 분양가격 동향_20200430.csv', delimiter=',', encoding='CP949')
    print(df_00)

    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


# pandas_temp()