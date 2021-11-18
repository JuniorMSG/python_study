"""
    subject
        Machine_Running
    topic
        금융데이터 분석
    Describe
        금융데이터 분석, 시계열 데이터,

        AR 모델 : AutoRegressive(자기회귀) Model
            자신의 바로 전 데이터를 활용하여 예측한다.

        시계열 데이터 생긴 모양
            Trend, Cycle을 분리하여 이해해야 한다.
    Contens
        01.
"""

import os
import seaborn as sns
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 절대경로 얻기
path = os.path.dirname(os.path.abspath(__file__))
folder_path = path + os.sep
folder_img_path = folder_path + 'image' + os.sep + '02_EX08_finance_data' + os.sep
img_li = os.listdir(folder_img_path)


def save_img_fig(fig, file_name):

    if str(type(fig)) == "<class 'matplotlib.figure.Figure'>":
        fig.savefig(folder_img_path + file_name)
    else:
        fig.get_figure().savefig(folder_img_path + file_name)
    plt.show()
    return


def finance_01():
    """
        subject
            Machine_Running
        topic
            금융데이터 분석
        content
            01. 시계열 데이터 Trend 및 Cycle 분해 및 시각화
        Describe
            Monthly Data
                호주 당뇨병 치료약(anti-diabetic) 월별 Sales 데이터 사용 예정 https://raw.githubusercontent.com/selva86/datasets/master/a10.csv
                모든 회사에 있는 월별 매출, 가입자, 등 실적 데이터에 활용
        sub Contents
            01.
    """

    print("\n", "=" * 5, "01", "=" * 5)
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
    print(df.sample(3))


    print("\n", "=" * 3, "01.", "=" * 3)
    file_name = '01_01_plot_chart.png'
    if file_name not in img_li:
        fig = df.value.plot()
        save_img_fig(fig, file_name)

    df = df[df.index > '1999-12-31']
    print(df.shape)

    file_name = '01_02_seasonal_decompose_plot.png'
    if file_name not in img_li:
        result = seasonal_decompose(df, model='additive', two_sided=False)
        fig = result.plot()
        save_img_fig(fig, file_name)

    df_re = pd.concat([result.observed, result.trend, result.seasonal, result.resid], axis=1)
    df_re.columns = ['obs', 'trend', 'seasonal', 'resid']
    df_re.dropna(inplace=True)
    df_re.head(24)
    df_re['year'] = df_re.index.year
    print(df_re.head())

    file_name = '01_03_seasonal_decompose_sum.png'
    if file_name not in img_li:
        fig = plt.figure(figsize=(16, 6))
        plt.plot(df_re.obs)
        plt.plot(df_re.trend)
        plt.plot(df_re.seasonal + df_re.trend)
        plt.legend(['observed', 'trend', 'seasonal+trend'])
        save_img_fig(fig, file_name)
    # 2006년 2007년에는 ordinary한 cycle에서 벗어나는 경향이 있다. (residual 크다. )

    print(df_re.index[0])

    def get_date(date):
        return (str(date.year) + '-' + str(date.month))

    print(get_date(df_re.index[0]))

    # trend
    file_name = '01_04_Trend.png'
    if file_name not in img_li:
        df_trend = df_re.trend.pct_change().dropna()
        ax = df_trend.plot(kind='bar', figsize=(16, 10))
        ax.set_xticklabels(list(map(lambda x: get_date(x), df_trend.index)))
        save_img_fig(ax, file_name)

    # residual : unexpected 값들이다.
    file_name = '01_05_residual.png'
    if file_name not in img_li:
        fig = df_re.groupby('year')['resid'].mean().plot(kind='bar')
        save_img_fig(fig, file_name)



    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


# finance_01()


def finance_02():
    """
        subject
            Machine_Running
        topic
            금융데이터 분석
        content
            02. 시계열 데이터 분석
        Describe
            시계열 데이터 특성
                안정성 (Stationary)
                    - 시계열 데이터가 미래에 똑같은 모양일 확률이 높다.
                    - 시계열이 안정적이지 않으면 현재의 패턴이 미래에 재현되지 않으므로 사용할 수 없다.
                불안정한 시계열을 그대로 예측에 활용할 경우
                    설명도, 정확도가 굉장히 높게 나오는데 잘못된 방법이다.
                    Spurious regression (가성적 회귀), Overfitting
                    원계열을 그대로 안쓰고 변화시켜서 사용한다.

                (Augmented) Dickey Fuller Test
                - 귀무가설 (Null Hypothesis) 원계열은 안정적이지 않다.

                테스트 결과
                    p-value 0.05보다 작으면 ,귀무가설 기각 - 안정적인 시계열
                    p-value 0.05보다 크면, 귀무가설 채택, - 불안정한 시계열
                    p-value란?
                        결과값이 특정 법칙을 따르는 것인지를 의미함
                        p-value가 작으면 특정 법칙을 따르는 것이고
                        p-value가 크면 우연히 결과가 나온것임.


        sub Contents
            01.
    """

    # https://fred.stlouisfed.org/
    df = pd.read_csv('data_file/DEXKOUS.csv', parse_dates=['DATE'], index_col='DATE')
    print(df.info())
    df.columns = ['KOUS']

    # 결측치 제거
    df['KOUS'].replace('.', '', inplace=True)
    df['KOUS'] = pd.to_numeric(df['KOUS'])
    df['KOUS'].fillna(method='ffill', inplace=True)

    file_name = '02_01_lineplot'
    if file_name not in img_li:
        fig = df['KOUS'].plot(figsize=(10, 6))
        save_img_fig(fig, file_name)

    #  resample : 일별데이터 -> 주단위 데이터, 월단위 데이터로 변환
    # df.resample('M').last()
    file_name = '02_02_lineplot_week.png'
    if file_name not in img_li:
        fig = df.resample('W-Fri').last().plot(figsize=(15, 6))
        save_img_fig(fig, file_name)

    # rolling : 이전 xx일에 대한 이동평균, 이동 sum 을 산출할 때 사용
    file_name = '02_03_rolling.png'
    if file_name not in img_li:
        fig = df.rolling(10).mean().plot(figsize=(16, 5))
        save_img_fig(fig, file_name)

    # rolling : 이전 xx일에 대한 이동평균, 이동 sum 을 산출할 때 사용
    file_name = '02_04_rolling_month.png'
    if file_name not in img_li:
        fig = df.rolling(30).std().resample('M').mean().plot()
        save_img_fig(fig, file_name)

    print("\n", "=" * 5, "02", "=" * 5)
    from statsmodels.tsa.stattools import adfuller
    """
        안정성 검정 (ADF Test)
            귀무가설=안정적이지 않다
            p-value가 0.05보다 작으면, 귀무가설 기각. 즉, 안정적인 시계열
            P-value가 0.05보다 크면, 귀무가설 채택. 즉, 불안정한 시계열
            안정적인 데이터로 변경 : 변화율 / 로그 차분
    """

    # y(t+1)/y(t) -1
    # log(y(t+1))-log(y(t))

    # df.KOUS.pct_change().dropna()
    print(adfuller(df['KOUS']))
    print(adfuller(df.KOUS.pct_change().dropna()))
    print(adfuller((df.KOUS / df.KOUS.shift(1) - 1).dropna()))
    print(adfuller((np.log(df.KOUS) - np.log(df.KOUS.shift(1))).dropna()))

    print("\n", "=" * 3, "01.", "=" * 3)




    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


finance_02()



def finance_03():
    """
        subject
            Machine_Running
        topic
            금융데이터 분석
        content
            03. 시계열의 두가지 유형 AR, MA
        Describe

            AR = Auto Regressive
                1. 이번 결과는 이전 결과에 영향을 받는 모델
                2. 외부 충격이 길게 반영되는 Long Memory model
            MA = 이동평균(Moving Average)
                1. 이번 결과는 이전 결과와 상관이 없음
                2. 외부 충격이 일정기간만 지속되고 없어지는 Short memory 모델

            ACF (Auto Correlation Function)
            PACF (Partial Auto Correlation Function)

        sub Contents
            01.
    """
    df = pd.read_csv('data_file/DEXKOUS.csv', parse_dates=['DATE'], index_col='DATE')
    print(df.info())
    df.columns = ['KOUS']

    # 결측치 제거
    df['KOUS'].replace('.', '', inplace=True)
    df['KOUS'] = pd.to_numeric(df['KOUS'])
    df['KOUS'].fillna(method='ffill', inplace=True)

    df_w = df.resample('W-Fri').last()
    df_2017 = df_w[df_w.index.year == 2017]
    df_2018 = df_w[df_w.index.year == 2018]
    df_2019 = df_w[df_w.index.year == 2019]
    df_2020 = df_w[df_w.index.year == 2020]
    print(df_2017.head())
    print(df_2018.head())
    print(df_2019.head())
    print(df_2020.head())

    print("\n", "=" * 5, "03", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)


    file_name = '03_01_year plot.png'
    if file_name not in img_li:
        fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='view')

        axes[0, 0].set_title('2017')
        df_2017.plot(ax=axes[0, 0])
        axes[0, 1].set_title('2018')
        df_2018.plot(ax=axes[0, 1])
        axes[1, 0].set_title('2019')
        df_2019.plot(ax=axes[1, 0])
        axes[1, 1].set_title('2020')
        df_2020.plot(ax=axes[1, 1])
        save_img_fig(fig, file_name)

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    file_name = '03_02_acf plot.png'
    if file_name not in img_li:
        fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='view')

        axes[0, 0].set_title('2017')
        axes[0, 1].set_title('2018')
        axes[1, 0].set_title('2019')
        axes[1, 1].set_title('2020')
        plot_acf(df_2017, ax=axes[0, 0])
        plot_acf(df_2018, ax=axes[0, 1])
        plot_acf(df_2019, ax=axes[1, 0])
        plot_acf(df_2020, ax=axes[1, 1])
        save_img_fig(fig, file_name)


    print("\n", "=" * 3, "02.", "=" * 3)

    # 첫번째 행 : 2017년 데이터의 원계열, ACF, PACF
    # 두번째 행 : 2019년 데이터의 원계열, ACF, PACF

    """
        2017에비해 2019년은 외부 충격이 오래 지속되었다. 3주 -4주
        2017년에는 외부충격이 다음기에 0.75 남아있지만, 2019년에는 0.9남아있다. (persistency가 증가하고 있다. )
        -> 가입자, 사용자 마케팅효과 분석
        -> 주가지수, 환율 : 외부 충격이 얼마나 오래 지속되는가.
    """
    file_name = '03_03_원계열 , ACF, PACF.png'
    if file_name not in img_li:

        fig, axes = plt.subplots(2, 3, figsize=(16, 7))
        axes[0, 0].plot(df_2017)
        axes[0, 0].set_title('original series(2017)')
        axes[1, 0].plot(df_2019)
        axes[1, 0].set_title('original series(2019)')
        plot_acf(df_2017, ax=axes[0, 1])
        plot_acf(df_2019, ax=axes[1, 1])
        plot_pacf(df_2017, ax=axes[0, 2])
        plot_pacf(df_2019, ax=axes[1, 2])
        save_img_fig(fig, file_name)


    print("\n", "=" * 3, "03.", "=" * 3)
    df = df[(df.index > '2019-01-01') & (df.index < '2020-01-01')]

    # ARIMA(p,k,q) => k 결정
    print(adfuller(df.KOUS))
    #  p-value : 0.36357542996557135 - 안정적이지 않은 시계열

    print(adfuller(df.KOUS.diff().dropna()))
    # p-value : 0 - 안정적인 시계열

    # 2x3 subplot

    # ARIMA(p,k,q) => p, q 결정
    file_name = '03_04_ARIMA(p,k,q).png'
    if file_name not in img_li:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        axes[0, 0].plot(df.KOUS)
        axes[0, 0].set_title('original series')
        axes[1, 0].plot(df.KOUS.diff())
        axes[1, 0].set_title('1st difference series')
        plot_acf(df.KOUS, ax=axes[0, 1])
        plot_pacf(df.KOUS, ax=axes[0, 2])
        plot_acf(df.KOUS.diff().dropna(), ax=axes[1, 1])
        plot_pacf(df.KOUS.diff().dropna(), ax=axes[1, 2])
        plt.tight_layout()
        save_img_fig(fig, file_name)

    # ARIMA 예측 모델링
    # ARIMA의 차수는 (3,1,2) -> (2,1,2)

    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(df.KOUS, order=(2, 1, 2), freq='B')
    model_fit = model.fit(trend='nc')
    print(model_fit.summary())

    # 의미없는 예측임.
    file_name = '03_05_의미없는 예측.png'
    if file_name not in img_li:
        fig = model_fit.plot_predict()
        save_img_fig(fig, file_name)

    # Training set, Test set을 나누어서 학습과 평가
    # Arima 데이터는 너무 길게 잡으면 옛날 데이터가 효과가 없기 때문에..
    # 짧게 예측만 가능하다..

    train = df.iloc[0:30]
    test = df.iloc[30:35]
    print(test.shape)

    model = ARIMA(train, order=(2, 1, 2), freq='B')
    model_fit = model.fit(trend='nc')
    fc, se, conf = model_fit.forecast(test.size, alpha=0.05)
    print(fc)
    print(se)
    print(conf)
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)


    # 03_06_ARIMA 5Day
    file_name = '03_06_ARIMA 5Day.png'
    if file_name not in img_li:
        fig = plt.figure(figsize=(13, 5))
        plt.plot(train, label='training')
        plt.plot(test, label='actual')
        plt.plot(fc_series, label='forecast')
        plt.fill_between(test.index, lower_series, upper_series, color='black', alpha=0.1)
        plt.legend(loc='upper left')
        save_img_fig(fig, file_name)


# finance_03()


def finance_04():
    """
        subject
            Machine_Running
        topic
            금융데이터 분석
        content
            04. 주식 종목 분석하기
        Describe
            금액이 클수록 리스크가 적은 자산이 우월하다.
            샤프 레이시오
                주식 30%       : 변동성 0.1 - 0.3 / 0.1 = 3
                비트코인 30%    : 변동성 0.5 - 0.3 / 0.5 = 0.6
                5배 차이
            예상 (초과) 수익률
                안전 자산 대비 얼마나 추가적인 수익률을 얻을 수 있는가 ? (ex 미국 국채)
            위험
                - 자산 가격이 얼마나 변동성이 있는가? -> 시계열의 표준 편차
            Sharpe Ratio
                - 예상 초과 수익률 / 위험

            1. 수익률 (Return)

            2. 리스크는 Rturn의 표준편차로 나타낼 수 있다.
            3. Sharpe ratio : risk adjusted return (단위 리스크당 수익)

            포트폴리오를 어떻게 하는지에 따라서 총 수익과 변동성이 달라집니다.
            - 주식 종목의 Return, Risk, Sharpe ratio
            - 포트폴리오의 REturn, Risk, Sharpe ratio
            - 포트폴리오 평가하기


            주식 정보 가져오기
                yahoo finance API 활용
                https://pypi.org/project/yfinance/
            종목코드 예시
                미국기업 : 'MSFT'
                한국기업 : '005930.KS' (삼성)
                인덱스 : '^KS11'
            종목코드 (Ticker symbol) 조회하기
                https://finance.yahoo.com/
                https://finance.naver.com/
                http://kind.krx.co.kr/corpgeneral/corpList.do?method=download (한국거래소)
       sub Contents
            01.
    """

    import yfinance as yf

    """
        포트폴리오 평가
            여러 조합으로 자산 포트폴리오를 구성한 뒤에 Return, Risk, Sharpe Ratio 비교 평가
            미국의 Tech기업, 한국의 Tech기업의 투자 포트폴리오 비교해보기
    """

    print("\n", "=" * 5, "04", "=" * 5)
    Tech_US = ['MSFT', 'NFLX', 'FB', 'AMZN']  # 마이크로소프트 , 넷플릭스, 페이스북, 아마존
    Tech_KR = ['005930.KS', '000660.KS', '035420.KS', '035720.KS']  # 삼성, SK하이닉스, 네이버, 카카오
    yf.Ticker('MSFT').history(start='2019-01-01', end='2021-07-30')

    print("\n", "=" * 3, "01.", "=" * 3)

    # price, dividends를 가져오는 함수를 정의
    def get_price(companies):
        df = pd.DataFrame()
        for company in companies:
            df[company] = yf.Ticker(company).history(start='2019-01-01', end='2021-07-30')['Close']
        return df

    def get_div(companies):
        df = pd.DataFrame()
        for company in companies:
            df[company] = yf.Ticker(company).history(start='2019-01-01', end='2021-07-30')['Dividends']
        return df

    # US, KR 테크 기업의 주식 가격
    p_US = get_price(Tech_US)
    p_KR = get_price(Tech_KR)
    p_KR.columns = ['SS', 'SKH', 'NVR', 'KKO']
    # US, KR 테크기업의 배당금
    d_US = get_div(Tech_US)
    d_KR = get_div(Tech_KR)

    d_KR.columns = ['SS', 'SKH', 'NVR', 'KKO']
    print(d_US.sum())
    print(d_KR.sum())

    # plotting
    (p_US / p_US.iloc[0]).plot(figsize=(15, 5))
    (p_KR / p_KR.iloc[0]).plot(figsize=(15, 5))

    print("\n", "=" * 3, "02.", "=" * 3)

    # Daily Return
    r_US = p_US/p_US.shift()-1
    r_KR = p_KR/p_KR.shift()-1

    # Average Return (Total period)
    r_a_US = (p_US.iloc[-1] + d_US.sum())/p_US.iloc[0] - 1
    r_a_KR = (p_KR.iloc[-1] + d_KR.sum())/p_KR.iloc[0] - 1

    # Averate Return (Daily)
    r_a_d_US = (1+r_a_US)**(1/p_US.shape[0])-1
    r_a_d_KR = (1+r_a_KR)**(1/p_KR.shape[0])-1


    # Portfolio Returns
    # weights 0.25, 0.25, 0.25, 0.25 가정
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    print(weights)

    port_return_US = np.dot(weights, r_a_US)
    port_return_KR = np.dot(weights, r_a_KR)

    # Portfolio Risk 낮을 수록 좋음
    covar_US = r_US.cov()*252
    covar_KR = r_KR.cov()*252
    print('1년 영업일이 252이므로 252 곱함.. ', covar_US)
    print('1년 영업일이 252이므로 252 곱함.. ', covar_KR)  # 1년 영업일 252/
    sns.heatmap(covar_US, cmap='PuBuGn') # 1년 영업일 252/
    sns.heatmap(covar_KR, cmap='PuBuGn')

    port_risk_US = np.dot(weights.T, np.dot(covar_US, weights))
    port_risk_KR = np.dot(weights.T, np.dot(covar_KR, weights))

    print('포트폴리오 리스크 US', port_risk_US)
    print('포트폴리오 리스크 KR', port_risk_KR)

    # Sharpe Ratio (단위 리스크당 수익률)
    # 미국 국채를 2%로 잡고
    rf = 0.02
    port_sr_US = (port_return_US - rf) / port_risk_US
    port_sr_KR = (port_return_KR - rf) / port_risk_KR
    print(port_sr_US)
    print(port_sr_KR)

    print("\n", "=" * 3, "03.", "=" * 3)

    # 시각화
    #                   KR, US
    # Return
    # Risk
    # Sharpe ration

    result = np.array([[port_return_KR, port_return_US], [port_risk_KR, port_risk_US], [port_sr_KR, port_sr_US]])
    result = np.round(result, 3)

    result = pd.DataFrame(result)
    result.columns = ['KR', 'US']
    result.index = ['Return', 'Risk', 'Sharpe ratio']
    print(result)

    result.plot(kind='bar')

    """
        나에게 있는 1억을 어떻게 분배 할 것인가.? 
        
        포트폴리오 최적화
        Portfolid Theory
        
        투자 상품의 수익과 리스크
        Efficient Frontier (효율적인 투자선?)
        

            
            
        
   """




finance_04()


def finance_05():
    """
        subject
            Machine_Running
        topic
            금융데이터 분석
        content
            05. 최적화 기초 개념
        Describe
            투자 포트폴리오 p
                목적함수 (Objective Function)
                    최대 또는 최소로 달성하고자 하는바
                    EX) 매출
                선택변수 (Choice variable)
                    목적함수를 최대 또는 최소로 만들기 위하여 선택 가능한 변수
                    EX) X1상품, X2상품으 ㅣ생산량
                제약조건 (Constraint condition)
                    목적함수를 달성하기 위하여 주어진 조건들
                    EX) X1상품과 X2상품을 생산하는데 소요되는 재료
                경계조건 (Boundary condition)
                EX) X1, X2는 0보다 작을 수 없음

        sub Contents
            01.
    """
    import yfinance as yf
    from scipy.optimize import minimize

    print("\n", "=" * 5, "05", "=" * 5)

    Tech_KR = ['005930.KS', '000660.KS', '035420.KS', '035720.KS']  # 삼성, SK하이닉스, 네이버, 카카오

    def get_price(companies):
        df = pd.DataFrame()
        for company in companies:
            df[company] = yf.Ticker(company).history(start='2020-07-01', end='2021-07-31')['Close']
        return df

    def get_div(companies):
        df = pd.DataFrame()
        for company in companies:
            df[company] = yf.Ticker(company).history(start='2020-07-01', end='2021-07-31')['Dividends']
        return df

    p_KR = get_price(Tech_KR)
    d_KR = get_div(Tech_KR)

    p_KR.columns = ['SS', 'SKH', 'NVR', 'KKO']
    d_KR.columns = ['SS', 'SKH', 'NVR', 'KKO']

    # 포트폴리오가 어떻게 생겼는가?
    # 3000개의 임의의 weights를 생성해서 return, risk를 도시
    weights = np.random.rand(len(Tech_KR))
    # weight는 1이 되야함
    weights = weights / np.sum(weights)
    print(weights)

    # 임의의 생성된 난수에 대한 Return
    r_a = (p_KR.iloc[-1] + d_KR.sum()) / p_KR.iloc[0] - 1
    port_return = np.dot(weights, r_a)
    print(port_return)

    # 임의의 생성된 난수에 대한 Risk
    covar_KR = (p_KR / p_KR.shift() - 1).cov() * 252
    port_risk = np.dot(weights.T, np.dot(covar_KR, weights))
    print(port_risk)

    # weights의 조합에 따른 포트폴리오 리턴, 리스크
    port_returns = []
    port_risks = []
    for ii in range(3000):
        weights = np.random.rand(len(Tech_KR))
        weights = weights / np.sum(weights)
        r_a = (p_KR.iloc[-1] + d_KR.sum()) / p_KR.iloc[0] - 1
        port_return = np.dot(weights, r_a)
        covar_KR = (p_KR / p_KR.shift() - 1).cov() * 252
        port_risk = np.dot(weights.T, np.dot(covar_KR, weights))
        port_returns.append(port_return)
        port_risks.append(port_risk)

    port_returns = np.array(port_returns)
    port_risks = np.array(port_risks)

    plt.scatter(port_risks, port_returns, c=port_returns / port_risks)
    plt.colorbar(label='Sharpe ratio')
    plt.xlabel('expected_risk')
    plt.ylabel('expected_return')
    plt.grid(True)
    plt.show()

    print("\n", "=" * 3, "01.", "=" * 3)
    """
        최적화 문제 정의
            목적함수 : Sharpe ratio(max), Risk (min)
            선택변수 : weights
            제약(constraint) : 모든 weights의 합은 1
            한계(boundary) : 각 weight는 0과 1 사이
        포트폴리오 최적화 : 세가지 점을 구해봄
            포트폴리오 리스크 최소
            Sharpe 지수 최대
            효율적 투자점 : 목표 수익을 달성하기 위한 최소 risk를 가질 수 있는 포트폴리오
            minimize(목적함수, w0, constraints=, bounds=)
    """

    # 목적함수 정의
    # 먼저, weights를 넣으면, return, risk, sharpe raio를 return하는 함수를 정의 => 목적함수 정의
    def get_stats(weights):
        r_a = (p_KR.iloc[-1] + d_KR.sum()) / p_KR.iloc[0] - 1
        port_return = np.dot(weights, r_a)
        covar_KR = (p_KR / p_KR.shift(1) - 1).cov() * 252
        port_risk = np.dot(weights.T, np.dot(covar_KR, weights))
        port_sharpe = port_return / port_risk
        return [port_return, port_risk, port_sharpe]

    def objective_return(weights):
        return -get_stats(weights)[0]

    def objective_risk(weights):
        return get_stats(weights)[1]

    def objective_sharpe(weights):
        return -get_stats(weights)[2]

    print(get_stats(weights))

    # w0 정의
    w0 = np.ones(len(Tech_KR)) / len(Tech_KR)

    # constraints
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # bounds
    bound = (0, 1)
    bounds = tuple(bound for ii in range(len(Tech_KR)))

    # 최적화 1. Risk 최소
    opt_risk = minimize(objective_risk, w0, constraints=constraints, bounds=bounds)

    # 최적화 2. Sharpe ration 최대
    opt_sharpe = minimize(objective_sharpe, w0, constraints=constraints, bounds=bounds)

    # 최적화된 risk
    print(opt_risk['fun'])
    # 그때의 weights (포트폴리오)
    print(opt_risk['x'])

    opt_risk['fun']  # 최적화된 risk
    opt_risk['x']  # 그때의 weights (포트폴리오)

    -opt_sharpe['fun']  # 최적화된 sharpe ratio
    opt_sharpe['x']  # 그때의 weights (포트폴리오)

    plt.scatter(port_risks, port_returns, c=port_returns / port_risks)
    plt.colorbar(label='Sharpe ratio')
    pt_opt_sharpe = get_stats(opt_sharpe['x'])
    plt.scatter(pt_opt_sharpe[1], pt_opt_sharpe[0], marker='*', s=500, c='black', alpha=0.5)
    pt_opt_risk = get_stats(opt_risk['x'])
    plt.scatter(pt_opt_risk[1], pt_opt_risk[0], marker='*', s=500, c='red', alpha=0.5)
    plt.xlabel('expected_risk')
    plt.ylabel('expected_return')
    plt.grid(True)
    plt.show()


    print("\n", "=" * 3, "02.", "=" * 3)

    # 효율적 투자점 : 목표 수익을 달성하기 위한 최소 risk를 가질 수 있는 포트폴리오
    target_returns = np.linspace(1.00, 4.00, 50)

    target_risks = []
    target_port = {}
    for target_return in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: get_stats(x)[0] - target_return})
        opt_target = minimize(objective_risk, w0, constraints=constraints, bounds=bounds)
        target_risks.append(opt_target['fun'])
        target_port[target_return] = opt_target['x']

    target_risks = np.array(target_risks)

    w = pd.DataFrame(target_port.values())
    w.columns = ['SS', 'SKH', 'NVR', 'KKO']
    w.index = target_returns.round(3)
    w.plot(figsize=(14, 6), kind='bar', stacked=True)

    plt.scatter(port_risks, port_returns, c=port_returns / port_risks)
    plt.colorbar(label='Sharpe ratio')
    plt.scatter(target_risks, target_returns, marker='x')
    plt.xlabel('expected_risk')
    plt.ylabel('expected_return')
    plt.grid(True)
    plt.show()


    print("\n", "=" * 3, "03.", "=" * 3)


finance_05()

def finance_temp():
    """
        subject
            Machine_Running
        topic
            금융데이터 분석
        content
            temp
        Describe

        sub Contents
            01.
    """
    print("\n", "=" * 5, "temp", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# finance_temp()
