"""
    subject
        Machine_Running
    topic
        마케팅 데이터 실습

    Describe


        axes : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
        x = np.linspace(0, 2 * np.pi, 400)
        y = np.sin(x ** 2)
        ax_temp.scatter(x, y)

    Contens
        01.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def marketing_01():
    """
        subject
            Machine_Running
        topic
            ex4. Kaggle_Data_Set : Advertising
        content
            01. 데이터 탐색
        Describe

            Kaggle Data_Set
                DATA
                TV      - TV 매체비
                Radio   - 라디오 매체비
                News    - 신문 매체비
                sales   - 매출액

                문제 정의
                    전제
                        실제로는 광고 매체비 이외의 많은 요인이 매출에 영향을 미친다. (영업인력 수 ,입소문, 경기, 유행 등..)
                        분석에서는 다른 요인이 모두 동일한 상황에서 매체비만 변경했을 때 매출액의 변화가 발생한 것이라고 간주
                        실제로 Acquisition 단계에서는 종속변수가 매출액보다는 방문자수, 가입자수, DAU, MAU등의 지표가 될 것.
                        2011년 데이터임
                    분석의 목적
                        각 미디어별로 매체비를 어떻게 쓰느냐에 따라서 매출액이 어떻게 달라질지 예측
                        궁극적으로는 매출액을 최대화 할 수 있는 미디어 믹스의 구성을 도출
                        이 미디어믹스는 향후 미디어 플랜을 수립할 때 사용 될 수 있다.

        sub Contents
            01.
    """
    # DATA                  - 재료
    # Processing Algorithm  - 조리법  (Python, 강화학습 등등..)
    # Application           - 요리    (알파고, 챗봇, 이런저런 어플리케이션 등등)
    # AARRR & (RARRA 중요도에 따라서 조금씩 다름)

    # Acquisition(사용자획득), Activation (사용자 활성화), Retention(사용자 유지), Revenue(매출), Referral(추천)

    # Activation(사용자 활성화)   : 사용자가 어떻게 서비스를 접하는가 ? ( DAU, MAU, New User, 방문자 수 등)
    # Acquisition(사용자획득)    : 사용자가 처음 서비스를 이용했을 때 경험이 좋았는가?  ( Avg, PV, Avg. Duration, 가입자 수 등)
    # Retention(사용자 유지) : 사용자가 우리 서비스를 계속 이용하는가? ( Retention Rate )
    # Revenue(매출) : 어떻게 돈을 버는가 ? ( Conversion )
    # Referral(추천) : 사용자가 다른 사람들에게 제품을 소개하는가? (SNS Share Rate)

    # STEP 1. 미디어별 광고비 EDA
    # STEP 2. 분석 모델링 매체비로 세일즈 예측
    # STEP 3. 분석 결과 해석 적용 방안

    print("\n", "=" * 5, "01", "=" * 5)
    df = pd.read_csv("./data_file/Advertising.csv")
    """
        기본 체크 작업
        1. 데이터 확인   : df.shape, df.head(), df.tail()
        2. 결측값 측정   : df.info(), df.isnull().sum()
        3. 분석에 필요한 컬럼만 선택
        4. 기술통계 확인 : df.describe()
        5. 변수간의 correlation 확인
        6. 변수간의 pairplot 확인
        7. Label, Feature(인풋 변수, 독립 변수) 지정
        
    """

    fig, axes = plt.subplots(2, 4, sharey=False, tight_layout=True, figsize=(15, 6), num='view')
    # fig = plt.figure(tight_layout=True, figsize=(15, 6), num='view')

    # 데이터 확인
    print(df.shape)
    print(df.tail())
    # 결측값 측정
    print(df.info())
    print(df.isnull().sum())

    # 분석에 필요한 컬럼만 선택
    df = df[['TV', 'radio', 'newspaper', 'sales']]

    # 기술통계 확인 : df.describe()
    print(df.describe())

    # 변수간의 correlation 확인 df.corr() 시각화
    corr = df.corr()
    print(corr)
    # annot=True 숫자 표시해줌

    ax_temp = axes[0, 0]
    ax_temp.set_title('HeatMap')
    sns.heatmap(corr, annot=True, ax=ax_temp)

    ax_temp = axes[0, 1]
    ax_temp.set_title('Scatter TV')
    sns.scatterplot(data=df, x='TV', y='sales', ax=ax_temp)

    ax_temp = axes[0, 2]
    ax_temp.set_title('Scatter Radio')
    sns.scatterplot(data=df, x='radio', y='sales', ax=ax_temp)

    ax_temp = axes[0, 3]
    ax_temp.set_title('Scatter News')
    sns.scatterplot(data=df, x='newspaper', y='sales', ax=ax_temp)




    # 6. 변수간의 pairplot 출력
    sns.pairplot(df[['TV', 'radio', 'newspaper', 'sales']])

    # 7. Label, Feature(인풋 변수, 독립 변수) 지정
    Labels = df['sales']
    features = df[['TV', 'radio', 'newspaper']]
    print(Labels.shape)
    print(features.shape)




    print("\n", "=" * 3, "01.", "=" * 3)

    # 선형회귀 분석 (stats model)
    import statsmodels.formula.api as sm
    model1 = sm.ols(formula='sales ~ TV + radio + newspaper', data=df).fit()
    print(model1.summary())

    # 분석이 잘된것인가 ?
    # R-squared: 0.897 높을 수록 좋지만 너무 높으면 무언가 잘못 된것..
    # P>|t| (P-value) 통계적으로 의미가 있는 값인가? (0.05 이상이면 유의하지 않다.)
    # coef

    # 선형회귀 분석 (sklearn model)
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    model2 = LinearRegression().fit(features, Labels)
    print(model2.intercept_ , model2.coef_)

    print("\n", "=" * 3, "02.", "=" * 3)

    model1 = sm.ols(formula='sales ~ TV + radio + newspaper', data=df).fit()
    model2 = sm.ols(formula='sales ~ TV + radio', data=df).fit()
    model3 = sm.ols(formula='sales ~ TV', data=df).fit()

    dict_data = {'TV' : 300, 'radio' : 10, 'newspaper' : 4}
    model1_pred = model1.predict({'TV' : 300, 'radio' : 10, 'newspaper' : 4})
    print(model1_pred)
    pred = 2.9389 + 0.0458 * dict_data['TV'] + 0.1885 * dict_data['radio'] - 0.0010 * dict_data['newspaper']
    print(pred)
    """
        Intercept      2.9389      0.312      9.422      0.000       2.324       3.554
        TV             0.0458      0.001     32.809      0.000       0.043       0.049
        radio          0.1885      0.009     21.893      0.000       0.172       0.206
        newspaper     -0.0010      0.006     -0.177      0.860      -0.013       0.011
    """

    print(model1.summary())
    print(model2.summary())
    print(model3.summary())

    # AIC, BIC가 가장 낮은 모델이 가장 좋은 모델로 판단 할 수 있는 기준이 된다.

    print("\n", "=" * 3, "03.", "=" * 3)

    # 데이터의 오류를 검증해보자
    # 미디어별 매체비 분포를 seaborn의 distplot으로 시각화\

    ax_temp = axes[1, 0]
    ax_temp.set_title('dist TV')
    sns.distplot(df['TV'], ax=ax_temp)

    ax_temp = axes[1, 1]
    ax_temp.set_title('dist radio')
    sns.distplot(df['radio'], ax=ax_temp)

    ax_temp = axes[1, 2]
    ax_temp.set_title('dist newspaper')
    sns.distplot(df['newspaper'], ax=ax_temp)


    df['log_newspaper'] = np.log(df['newspaper'] + 1)
    ax_temp = axes[1, 3]
    ax_temp.set_title('dist log_newspaper')
    sns.distplot(df['log_newspaper'], ax=ax_temp)
    print(df[['log_newspaper', 'newspaper']])

    # newspaper 값이 치우쳐져 있으니 정규화를 위해 로그 변환
    # 0이 되면 음의 무한대값이 되기때문에 + 숫자를 해줘서 방지하여 로그로 변경한다.

    model1 = sm.ols(formula='sales ~ TV + radio + newspaper', data=df).fit()
    # model2 = sm.ols(formula='sales ~ TV + radio', data=df).fit()
    # model3 = sm.ols(formula='sales ~ TV', data=df).fit()
    model4 = sm.ols(formula='sales ~ TV + radio + log_newspaper', data=df).fit()

    print(model1.summary())
    print(model4.summary())

    plt.show()


    # 적용 방안


    print(df.shape)


# marketing_01()


def marketing_02():
    """
        subject
            Machine_Running
        topic
            ex4. marketing_data
        content
            02. Retention
        Describe
            허상적 지표(Vanity Metric)
            행동적 지표 (Actionable Metric)

            OMTM - facebook,
            서비스에 본질에 가까운 분석이 필요함

            STEP 1. 모바일 게임 A/B Test 데이터
            STEP 2. 두 집단의 A/B Test Retention 비교
            STEP 3. 분석 결과 해석 적용 방안

        DATA_SET
            userid - 개별 유저들을 구분하는 식별 번호
            version - 유저들이 실험군 대조군 중 속한 위치
            sum_gamerounds - 첫 설치 후 14일 간 유저가 플레이한 라운드의 수
            retention_1 - 유저가 설치 후 1일 이내에 다시 돌아왔는지 여부
            retention_7 - 유저가 설치 후 7일 이내에 다시 돌아왔는지 여부

        ub Contents
            01.
    """

    print("\n", "=" * 5, "02", "=" * 5)
    df = pd.read_csv("./data_file/cookie_cats.csv")
    """
        기본 체크 작업
        1. 데이터 확인   : df.shape, df.head(), df.tail()
        2. 결측값 측정   : df.info(), df.isnull().sum()
        3. 분석에 필요한 컬럼만 선택
        4. 기술통계 확인 : df.describe()
        5. 변수간의 correlation 확인
        6. 변수간의 pairplot 확인
        7. Label, Feature(인풋 변수, 독립 변수) 지정
    """

    fig, axes = plt.subplots(2, 4, sharey=False, tight_layout=True, figsize=(15, 6), num='mobile game')

    # 데이터 확인
    print(df.shape)
    print(df.tail())
    # 결측값 측정
    print(df.info())
    print(df.isnull().sum())

    # 분석에 필요한 컬럼만 선택
    df = df[['userid', 'version', 'sum_gamerounds', 'retention_1', 'retention_7']]

    # 기술통계 확인 : df.describe()
    print(df.describe())

    # 변수간의 correlation 확인 df.corr() 시각화
    corr = df.corr()
    print(corr)
    # annot=True 숫자 표시해줌

    ax_temp = axes[0, 0]
    ax_temp.set_title('HeatMap')
    sns.heatmap(corr, annot=True, ax=ax_temp)

    ax_temp = axes[0, 1]
    ax_temp.set_title('sum_gamerounds box plot')
    sns.boxenplot(data=df, y='sum_gamerounds', ax=ax_temp)

    # 이상한 데이터 제거
    df[df['sum_gamerounds'] > 45000]
    df = df[df['sum_gamerounds'] < 45000]
    print(df['sum_gamerounds'].describe())

    ax_temp = axes[0, 2]
    ax_temp.set_title('sum_gamerounds box plot')
    sns.boxenplot(data=df, y='sum_gamerounds', ax=ax_temp)

    print("\n", "=" * 3, "01.", "=" * 3)
    # 데이터 분석시작.
    # 그룹 확인.
    print(df.groupby('version').count())

    # 게임 횟수별 유저수 확인
    plot_df = df.groupby('sum_gamerounds')['userid'].count()
    print(plot_df)

    ax_temp = axes[0, 3]
    ax_temp.set_title('Line plot')
    ax_temp.set_ylabel('Number of Player')
    ax_temp.set_xlabel('# Game rounds')
    plot_df[:300].plot(figsize=(10, 6), ax=ax_temp)

    ax_temp = axes[1, 0]
    ax_temp.set_title('distplot')
    sns.distplot(df['sum_gamerounds'], ax=ax_temp)

    # ax_temp = axes[1, 1]
    # ax_temp.set_title('Scatter Radio')
    # sns.scatterplot(data=df, x='retention_1', ax=ax_temp)
    #
    # ax_temp = axes[1, 2]
    # ax_temp.set_title('Scatter News')
    # sns.scatterplot(data=df, x='retention_7', ax=ax_temp)

    # 6. 변수간의 pairplot 출력
    # sns.pairplot(df[['TV', 'radio', 'newspaper', 'sales']])
    #
    # 7. Label, Feature(인풋 변수, 독립 변수) 지정
    # Labels = df['sales']
    # features = df[['TV', 'radio', 'newspaper']]
    # print(Labels.shape)
    # print(features.shape)


    # 1-day retention 정보 조회
    # 평균
    print('', df['retention_1'].mean())
    print('retention_1 평균', df['retention_1'].mean(), df.groupby('version')['retention_1'].mean())
    print('retention_7 평균', df['retention_7'].mean(), df.groupby('version')['retention_7'].mean())

    # 분석 결과 해석 적용 방안
    # 정말 데이터간의 차이가 있는것일까?

    # Bootstrap 방법 .
    # T-test 방법 (연속형 숫자일때 가능)
    # Chai Square 정말로 유의미하게 차이가 있는것인가 ?

    boot_1d = []
    for i in range(1000):
        boot_mean = df.sample(frac=1, replace=True).groupby('version')['retention_1'].mean()
        boot_1d.append(boot_mean)

    # list를 DataFrame으로 변환합니다.
    boot_1d = pd.DataFrame(boot_1d)

    # A Kernel Density Estimate plot of the bootstrap distributions
    boot_1d.plot(kind='density')
    plt.show()

    """
        위의 두 분포는 AB 두 그룹에 대해 1 day retention이 가질 수 있는 부트 스트랩 불확실성을 표현합니다.
        비록 작지만 차이의 증거가있는 것 같아 보입니다.
        자세히 살펴보기 위해 % 차이를 그려 봅시다.
    """

    # 두 AB 그룹간의 % 차이 평균 컬럼을 추가합니다.
    boot_1d['diff'] = (boot_1d.gate_30 - boot_1d.gate_40) / boot_1d.gate_40 * 100

    # bootstrap % 차이를 시각화 합니다.
    ax = boot_1d['diff'].plot(kind='density')
    ax.set_title('% difference in 1-day retention between the two AB-groups')

    # 게이트가 레벨30에 있을 때 1-day retention이 클 확률을 계산합니다.
    print('게이트가 레벨30에 있을 때 1-day retention이 클 확률:', (boot_1d['diff'] > 0).mean())
    """
        위 도표에서 가장 가능성이 높은 % 차이는 약 1%-2%이며 분포의 95%는 0% 이상이며 레벨 30의 게이트를 선호합니다.
        부트 스트랩 분석에 따르면 게이트가 레벨 30에있을 때 1일 유지율이 더 높을 가능성이 높습니다.
        그러나 플레이어는 하루 동안 만 게임을했기 때문에 대부분의 플레이어가 아직 레벨 30에 다다르지 않았을 가능성이 큽니다.
        즉, 대부분의 유저들은 게이트가 30에 있는지 여부에 따라 retention이 영향받지 않았을 것입니다.
        일주일 동안 플레이 한 후에는 더 많은 플레이어가 레벨 30과 40에 도달하기 때문에 7 일 retention도 확인해야합니다.
    """

    df.groupby('version')['retention_7'].sum() / df.groupby('version')['retention_7'].count()

    boot_7d = []
    for i in range(500):
        boot_mean = df.sample(frac=1, replace=True).groupby('version')['retention_7'].mean()
        boot_7d.append(boot_mean)

    # list를 DataFrame으로 변환합니다.
    boot_7d = pd.DataFrame(boot_7d)

    # 두 AB 그룹간의 % 차이 평균 컬럼을 추가합니다.
    boot_7d['diff'] = (boot_7d.gate_30 - boot_7d.gate_40) / boot_7d.gate_40 * 100

    # bootstrap % 차이를 시각화 합니다.
    ax = boot_7d['diff'].plot(kind='density')
    ax.set_title('% difference in 7-day retention between the two AB-groups')

    # 게이트가 레벨30에 있을 때 7-day retention이 더 클 확률을 계산합니다.
    print('게이트가 레벨30에 있을 때 7-day retention이 클 확률:', (boot_7d['diff'] > 0).mean())

    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    # T-test 통계적인 기준으로 판단하는 방법.
    """
        https://www.statisticshowto.com/probability-and-statistics/t-test/
        T Score
            t-score가 크면 두 그룹이 다르다는 것을 의미합니다.
            t-score가 작으면 두 그룹이 비슷하다는 것을 의미합니다.
        P-values
            p-value는 5%수준에서 0.05입니다.
            p-values는 작은 것이 좋습니다. 이것은 데이터가 우연히 발생한 것이 아니라는 것을 의미합니다.
            예를 들어 p-value가 0.01 이라는 것은 결과가 우연히 나올 확률이 1%에 불과하다는 것을 의미합니다.
            대부분의 경우 0.05 (5%) 수준의 p-value를 기준으로 삼습니다. 이 경우 통계적으로 유의하다고 합니다.
            
        위 분석결과를 보면, 두 그룹에서 retention_1에 있어서는 유의하지 않고, retention_7에서는 유의미한 차이가 있다는 것을 알 수 있습니다.
        다시말해, retention_7이 gate30이 gate40 보다 높은 것은 우연히 발생한 일이 아닙니다.
        즉, gate는 30에 있는 것이 40에 있는 것보다 retention 7 차원에서 더 좋은 선택지 입니다.
    """


    # 수학적 계산을 해주는 파이썬 패키지
    from scipy import stats

    df_30 = df[df['version'] == 'gate_30']
    df_40 = df[df['version'] == 'gate_40']

    # 독립표본 T-검정 (2 Sample T-Test)
    tTestResult = stats.ttest_ind(df_30['retention_1'], df_40['retention_1'])
    tTestResultDiffVar = stats.ttest_ind(df_30['retention_1'], df_40['retention_1'], equal_var=False)

    print(tTestResult, tTestResultDiffVar)

    tTestResult = stats.ttest_ind(df_30['retention_7'], df_40['retention_7'])
    tTestResultDiffVar = stats.ttest_ind(df_30['retention_7'], df_40['retention_7'], equal_var=False)
    print(tTestResult, tTestResultDiffVar)

    print("\n", "=" * 3, "03.", "=" * 3)

    # chi-square
    """
        chi-square
        사실 t-test는 retention 여부를 0,1 로 두고 분석한 것입니다.
        하지만 실제로 retention 여부는 범주형 변수입니다. 이 방법보다는 chi-square검정을 하는 것이 더 좋은 방법입니다.
        카이제곱검정은 어떤 범주형 확률변수 𝑋 가 다른 범주형 확률변수 𝑌 와 독립인지 상관관계를 가지는가를 검증하는데도 사용됩니다.
        카이제곱검정을 독립을 확인하는데 사용하면 카이제곱 독립검정이라고 부릅니다.
        만약 두 확률변수가 독립이라면 𝑋=0 일 때의 𝑌 분포와 𝑋=1 일 때의 𝑌 분포가 같아야 합니다.
        다시말해 버전이 30일때와 40일 때 모두 Y의 분포가 같은 것입니다.
        따라서 표본 집합이 같은 확률분포에서 나왔다는 것을 귀무가설로 하는 카이제곱검정을 하여 채택된다면 두 확률변수는 독립입니다.
        만약 기각된다면 두 확률변수는 상관관계가 있는 것입니다.
        다시말해 카이제곱검정 결과가 기각된다면 게이트가 30인지 40인지 여부에 따라 retention의 값이 변화하게 된다는 것입니다.
        𝑋 의 값에 따른 각각의 𝑌 분포가 2차원 표(contingency table)의 형태로 주어지면 독립인 경우의 분포와 실제 y 표본본포의 차이를 검정통계량으로 계산합니다.
        이 값이 충분히 크다면 𝑋 와 𝑌 는 상관관계가 있다.
    """
    # 분할표를 만들기 위해 버전별로 생존자의 수 합계를 구합니다.
    df.groupby('version').sum()

    # 버전별 전체 유저의 수를 구합니다.
    df.groupby('version').count()

    import scipy as sp
    gate_40_01_sum = df.groupby('version').sum()['retention_1']['gate_40']
    gate_30_01_sum = df.groupby('version').sum()['retention_1']['gate_30']
    gate_40_07_sum = df.groupby('version').sum()['retention_7']['gate_40']
    gate_30_07_sum = df.groupby('version').sum()['retention_7']['gate_30']

    gate_40_01_cnt = df.groupby('version').count()['retention_1']['gate_40']
    gate_30_01_cnt = df.groupby('version').count()['retention_1']['gate_30']
    gate_40_07_cnt = df.groupby('version').count()['retention_7']['gate_40']
    gate_30_07_cnt = df.groupby('version').count()['retention_7']['gate_30']


    obs1 = np.array([[gate_40_01_sum, (gate_40_01_cnt - gate_40_01_sum)], [gate_30_01_sum, (gate_30_01_cnt - gate_30_01_sum)]])
    sp.stats.chi2_contingency(obs1)

    obs2 = np.array([[gate_40_07_sum, (gate_40_07_cnt - gate_40_07_sum)], [gate_30_07_sum, (gate_30_07_cnt - gate_30_07_sum)]])
    sp.stats.chi2_contingency(obs2)

    """
        OBS-1
        P-value가 중요 (0.075)
        (3.1698355431707994,
         0.07500999897705699,
         1,
         array([[20252.35970417, 25236.64029583],
                [19900.64029583, 24798.35970417]]))
        OBS-2
        P-value가 중요 (0.001)
        (9.915275528905669,
         0.0016391259678654423,
         1,
         array([[ 8463.49203885, 37025.50796115],
                [ 8316.50796115, 36382.49203885]]))
        카이제곱 독립검정의 유의확률은 0.1%입니다.
        즉 𝑋 와 𝑌 는 상관관계가 있다고 말할 수 있습니다.
        게이트가 30에 있는지 40에 있는지 여부에 따라 7일 뒤 retention이 상관관계가 있는 것입니다.
        7일 뒤 retention 유지를 위하여 게이트는 30에 유지해야 합니다.
    """

    """
        Bootstrap 
        T-Test  : 크고 작은것에 의미가 있는 데이터일 경우 ..? 
        Chai Square 데이터가 True False로 나올때..
        
        결론
            gate는 30에 유지해야합니다.
            
            더 생각해 볼 것
            실제로는 retention 이외에 함께 고려해야 할 다양한 메트릭들이 있습니다.
            앱내 구매, 게임 플레이 횟수, 친구초대로 인한 referrer 등 입니다.
            본 데이터에서는 retention만 주어져 있기에 한 가지를 주안점을 두어 분석 했습니다.
            서비스 운영자, 기획자 차원에서 정말 중요한 메트릭을 정하고 그 것을 기준으로 테스트 결과를 평가하는 것이 중요합니다.
    """


# marketing_02()


def marketing_03():
    """
        subject
            Machine_Running
        topic
            ex4. marketing_data
        content
            03. Revenue 고객 세그먼트를 나눠보자
        Describe
            분석할 데이터 파악
            mall 데이터 EDA
                CustomerID - 고객들에게 배정된 유니크한 고객 번호 입니다.
                Gender - 고객의 성별 입니다.
                Age - 고객의 나이 입니다.
                Annual Income (k$) - 고객의 연소득 입니다.
                Spending Score (1-100) - 고객의 구매행위와 구매 특성을 바탕으로 mall에서 할당한 고객의 지불 점수 입니다.

            문제 정의
                전제
                    주어진 데이터가 적절 정확하게 수집, 계산된 것인지에 대한 검증부터 시작해야하지만,
                    지금은 주어진 데이터가 정확하다고 가정합니다.
                    (예: Spending Score는 적절하게 산출된 것이라 확신하고 시작합니다)
                    주어진 변수들을 가지고 고객 세그먼트를 도출합니다.
                    가장 적절한 수의 고객 세그먼트를 도출합니다.
                분석의 목적
                    각 세그먼트 별 특성을 도출합니다.
                    각 세그먼트별 특성에 맞는 활용방안, 전략을 고민해봅니다.
        sub Contents
            01.
    """

    df = pd.read_csv("./data_file/Mall_Customers.csv")
    fig, axes = plt.subplots(2, 4, sharey=False, tight_layout=True, figsize=(15, 6), num='mobile game')

    # 데이터 확인
    print(df.shape)
    print(df.tail())

    # 결측값 측정
    print(df.info())
    print(df.isnull().sum())

    # 기술통계 확인 : df.describe()
    print(df.describe())

    # 변수간의 correlation 확인 df.corr() 시각화
    def open_data_graph():
        corr = df.corr()
        print(corr)

        ax_temp = axes[0, 0]
        ax_temp.set_title('HeatMap')
        sns.heatmap(corr, annot=True, ax=ax_temp)

        # pairplot 시각화 생성
        print(df.columns)

        ax_temp = axes[0, 1]
        ax_temp.set_title('Age dist plot')
        sns.distplot(df['Age'], ax=ax_temp)

        ax_temp = axes[0, 2]
        ax_temp.set_title('Annual Income (k$) dist plot')
        sns.distplot(df['Age'], ax=ax_temp)

        ax_temp = axes[0, 3]
        ax_temp.set_title('Spending Score (1-100) dist plot')
        sns.distplot(df['Spending Score (1-100)'], ax=ax_temp)

        ax_temp = axes[1, 0]
        ax_temp.set_title('Genter Count plot')
        sns.countplot(df['Gender'], ax=ax_temp)

        ax_temp = axes[1, 1]
        ax_temp.set_title('boxplot plot')
        sns.boxplot(data=df, x='Gender', y='Age', hue='Gender', palette=['m', 'g'], ax=ax_temp)


        # 서브플롯에 안들어가는 것들
        sns.pairplot(df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
        sns.pairplot(df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']], hue='Gender')
        # lmplot은 regplot을 서브 플롯으로 하는 플롯입니다
        sns.lmplot(data=df, x='Age', y='Annual Income (k$)', hue='Gender', fit_reg=False)
        sns.lmplot(data=df, x='Spending Score (1-100)', y='Annual Income (k$)', hue='Gender', fit_reg=False)

        plt.show()

    # open_data_graph()


    print("\n", "=" * 5, "03", "=" * 5)

    print("\n", "=" * 3, "01.", "=" * 3)
    """
        고객 세그먼트 클러스터링
        K-means를 사용한 클러스터링
            K-means는 가장 빠르고 단순한 클러스터링 방법 중 한 가지 입니다.
            scikit-learn의 cluster 서브패키지 KMeans 클래스를 사용합니다. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        
            n_clusters: 군집의 갯수 (default=8)
            init: 초기화 방법. "random"이면 무작위, "k-means++"이면 K-평균++ 방법.(default=k-means++)
            n_init: centroid seed 시도 횟수. 무작위 중심위치 목록 중 가장 좋은 값을 선택한다.(default=10)
            max_iter: 최대 반복 횟수.(default=300)
            random_state: 시드값.(default=None)     
        
        2가지 변수가 아닌 여러가지 변수를 활용한후 차원 축소를 통해서 군집화 한다.
    """
    ### Age & spending Score 두 가지 변수를 사용한 클러스터링
    # ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']


    from sklearn.cluster import KMeans
    # X1에 'Age' , 'Spending Score (1-100)'의 값을 넣어줍니다.
    x1 = df[['Age', 'Spending Score (1-100)']].values
    print(x1.shape)

    # inertia 라는 빈 리스트를 만들어줍니다.
    inertia = []

    # 군집수 n을 1에서 20까지 돌아가며 X1에 대해 k-means++ 알고리즘을 적용하여 inertia를 리스트에 저장합니다.
    for n in range(1, 20):
        algorithm = (KMeans(n_clusters=n, random_state=30))
        algorithm.fit(x1)
        inertia.append(algorithm.inertia_)

    """
        Inertia value를 이용한 적정 k 선택
        관성(Inertia)에 기반하여 n 개수를 선택합니다.
        관성(Inertia) : 각 중심점(centroid)에서 군집 내 데이터간의 거리를 합산한 것으로 군집의 응집도를 나타냅니다. 
        이 값이 작을수록 응집도 높은 군집화 입니다. 즉, 작을 수록 좋은 값 입니다.
        https://scikit-learn.org/stable/modules/clustering.html
    """


    print(inertia)
    axes[0, 0].plot(np.arange(1, 20), inertia, 'o')
    axes[0, 0].plot(np.arange(1, 20), inertia, '-', alpha=0.8)
    axes[0, 0].set_title('KMeans small is best')
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0,0].set_ylabel('Inertia')
    plt.show()

    # 급격하게 변하는 지점이 군집으로 잡기에 좋다.
    # 군집수를 4로 지정하여 시각화 해봅니다.
    algorithm = (KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan'))
    algorithm.fit(x1)
    labels1 = algorithm.labels_
    centroids1 = algorithm.cluster_centers_

    h = 0.02
    x_min, x_max = x1[:, 0].min() - 1, x1[:, 0].max() + 1
    y_min, y_max = x1[:, 1].min() - 1, x1[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(1, figsize=(15, 7))
    plt.clf()
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

    plt.scatter(x='Age', y='Spending Score (1-100)', data=df, c=labels1,
                s=200)
    plt.scatter(x=centroids1[:, 0], y=centroids1[:, 1], s=300, c='red', alpha=0.5)
    plt.ylabel('Spending Score (1-100)'), plt.xlabel('Age')
    plt.show()

    """
        연령-소비점수를 활용한 군집 4개는 아래와 같이 명명할 수 있습니다.

        저연령-고소비 군
        저연령-중소비 군
        고연령-중소비 군
        저소비 군
        군집별 활용 전략 예시
            이 수퍼마켓 mall의 경우 소비점수가 높은 고객들은 모두 40세 이하의 젊은 고객입니다.
            소비점수가 높은 고객들은 연령대가 비슷한 만큼 비슷한 구매패턴과 취향을 가질 가능성이 높습니다.
            해당 군집의 소비자 특성을 더 분석해본 뒤 해당 군집의 소비자 대상 VIP 전략을 수립해봅니다.
            소비점수가 중간정도인 고객들에게는 연령에 따라 두 개 집단으로 나눠서 접근해봅니다.
            소비점수가 낮은 고객군은 연령대별로 중소비점수 군집에 편입될 수 있도록 접근해봅니다.
    """

    print("\n", "=" * 3, "02.", "=" * 3)
    # X1에 'Annual Income (k$)' , 'Spending Score (1-100)' 의 값을 넣어줍니다.
    X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

    # inertia 라는 빈 리스트를 만들어줍니다.
    inertia = []

    # 군집수 n을 1에서 11까지 돌아가며 X1에 대해 k-means++ 알고리즘을 적용하여 inertia를 리스트에 저장합니다.
    for n in range(1, 11):
        algorithm = (KMeans(n_clusters=n))
        algorithm.fit(X2)
        inertia.append(algorithm.inertia_)

    plt.figure(1, figsize=(16, 5))
    plt.plot(np.arange(1, 11), inertia, 'o')
    plt.plot(np.arange(1, 11), inertia, '-', alpha=0.8)
    plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
    plt.show()

    # 군집수를 5로 지정하여 시각화 해봅니다.
    algorithm = (KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan'))
    algorithm.fit(X2)
    labels2 = algorithm.labels_
    centroids2 = algorithm.cluster_centers_

    h = 0.02
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(1, figsize=(15, 7))
    plt.clf()
    Z2 = Z2.reshape(xx.shape)
    plt.imshow(Z2, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

    plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, c=labels2, s=200)
    plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=300, c='red', alpha=0.5)
    plt.ylabel('Spending Score (1-100)'), plt.xlabel('Annual Income (k$)')
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)

    """
        분석 모델링 / 고객 세그먼트 해석.
        실루엣 스코어를 사용한 k 선택
        가장 좋은 값은 1이고 최악의 값은 -1
       
            Silhouette Coefficient는 각 샘플의 클러스터 내부 거리의 평균 (a)와 인접 클러스터와의 거리 평균 (b)을 사용하여 계산합니다.
            한 샘플의 Silhouette Coefficient는 (b - a) / max(a, b)입니다.
            
            0 근처의 값은 클러스터가 오버랩되었다는 것을 의미합니다
            음수 값은 샘플이 잘못된 클러스터에 배정되었다는 것을 의미합니다. 다른 클러스터가 더 유사한 군집이라는 의미입니다.
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.cm as cm

    # 클러스터의 갯수 리스트를 만들어줍니다.
    range_n_clusters = [6]

    # 사용할 컬럼 값을 지정해줍니다.
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

    for n_clusters in range_n_clusters:
        # 1 X 2 의 서브플롯을 만듭니다.
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # 첫 번째 서브플롯은 실루엣 플롯입니다.
        # silhouette coefficient는 -1에서 1 사이의 값을 가집니다.
        # 하지만 시각화에서는 -0.1에서 1사이로 지정해줍니다.
        ax1.set_xlim([-0.1, 1])

        # clusterer를 n_clusters 값으로 초기화 해줍니다.
        # 재현성을 위해 random seed를 10으로 지정 합니다.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # silhouette_score는 모든 샘플에 대한 평균값을 제공합니다.
        # 실루엣 스코어는 형성된 군집에 대해 밀도(density)와 분리(seperation)에 대해 견해를 제공합니다.
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # 각 샘플에 대한 실루엣 스코어를 계산합니다.
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # 클러스터 i에 속한 샘플들의 실루엣 스코어를 취합하여 정렬합니다.
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # 각 클러스터의 이름을 달아서 실루엣 플롯의 Label을 지정해줍니다.
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # 다음 플롯을 위한 새로운 y_lower를 계산합니다.
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # 모든 값에 대한 실루엣 스코어의 평균을 수직선으로 그려줍니다.
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # yaxis labels / ticks 를 지워줍니다.
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 두 번째 플롯이 실제 클러스터가 어떻게 형성되었는지 시각화 합니다.
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # 클러스터의 이름을 지어줍니다.
        centers = clusterer.cluster_centers_
        # 클러스터의 중앙에 하얀 동그라미를 그려줍니다.
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()

    print(n_clusters, cluster_labels, cluster_labels.shape)
    df['cluster'] = cluster_labels
    print(df.groupby('cluster')['Age'].mean())
    sns.boxplot(x='cluster', y="Age", hue="Gender", palette=["c", "m"], data=df)
    """
        boxplot은 중앙값, 표준 편차 등, 분포의 간략한 특성을 보여줍니다.
        각 카테고리 값에 따른 분포의 실제 데이터와 형상을 보고 싶다면 violinplot, stripplot, swarmplot 등으로 시각화 해봅니다.
        violinplot은 세로 방향으로 커널 밀도 히스토그램을 그려줍니다. 양쪽이 왼쪽, 오른쪽 대칭이 되도록 하여 바이올린처럼 보입니다.
        violinplot: http://seaborn.pydata.org/generated/seaborn.violinplot.html
        swarmplot은 stripplot과 유사하며 데이터를 나타내는 점이 겹치지 않도록 옆으로 이동해서 그려줍니다.
        swarmplot: http://seaborn.pydata.org/generated/seaborn.swarmplot.html
    """

    df['cluster'] = cluster_labels
    print(df.tail())

    # 각 그룹의 특성을 확인하기
    print(df.groupby('cluster')['Age'].mean())

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='mobile game')
    # violinplot
    sns.violinplot(x='cluster', y='Annual Income (k$)', data=df, inner=None, ax=axes[0, 0])
    sns.swarmplot(x='cluster', y="Annual Income (k$)", data=df, ax=axes[0, 0], color='white', edgecolor='gray')
    sns.boxplot(x='cluster', y="Annual Income (k$)", data=df, ax=axes[0, 1])

    sns.violinplot(x='cluster', y='Spending Score (1-100)', data=df, inner=None, ax=axes[1, 0])
    sns.swarmplot(x='cluster', y="Spending Score (1-100)", data=df, ax=axes[1, 0], color='white', edgecolor='gray')

    sns.violinplot(x='cluster', y='Age', data=df, inner=None, ax=axes[1, 1])
    sns.swarmplot(x='cluster', y="Age", data=df, ax=axes[1, 1], color='white', edgecolor='gray')
    plt.show()

    # 3개의 시각화를 한 화면에 배치합니다.
    figure, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3)

    # 시각화의 사이즈를 설정해줍니다.
    figure.set_size_inches(20, 6)
    # 클러스터별로 swarmplot을 시각화해봅니다.
    ax1 = sns.violinplot(x="cluster", y='Annual Income (k$)', data=df, inner=None, ax=ax1)
    ax1 = sns.swarmplot(x="cluster", y='Annual Income (k$)', data=df,
                        color="white", edgecolor="gray", ax=ax1)

    ax2 = sns.violinplot(x="cluster", y='Spending Score (1-100)', data=df, inner=None, ax=ax2)
    ax2 = sns.swarmplot(x="cluster", y='Spending Score (1-100)', data=df,
                        color="white", edgecolor="gray", ax=ax2)

    ax3 = sns.violinplot(x="cluster", y='Age', data=df, inner=None, ax=ax3)
    ax3 = sns.swarmplot(x="cluster", y='Age', data=df,
                        color="white", edgecolor="gray", ax=ax3, hue="Gender")
    plt.show()
    """
       
        추가 분석을 해본다면
        "Gender" 변수 활용
        K-means는 기본적으로 numerical variable을 사용하는 알고리즘입니다. 유클리디안 거리를 계산해야하기 때문입니다.
        Gender 변수를 one-hot-encoding하여 숫자로 바꿔준 뒤 변수로 추가하여 활용해봅니다.
        
        카테고리 변수가 대부분인 경우의 군집화
        k-modes 알고리즘을 사용합니다.
        https://pypi.org/project/kmodes/
        
    """


# marketing_03()


def marketing_04():
    """
        subject
            Machine_Running
        topic
            ex4. marketing_data
        content
            04. Revenue
        Describe
            고객 해지율을 낮추고 CLV를 높여보자
        sub Contents
            01. 분석할 데이터 파악 ( 통신사 고객 데이터 EDA )

            해지 여부
                Churn - 고객이 지난 1개월 동안 해지했는지 여부 (Yes or No)

            Demographic 정보
                customerID - 고객들에게 배정된 유니크한 고객 번호 입니다.
                gender - 고객의 성별 입니다(male or a female).
                Age - 고객의 나이 입니다.
                SeniorCitizen - 고객이 senior 시민인지 여부(1, 0).
                Partner - 고객이 파트너가 있는지 여부(Yes, No).
                Dependents - 고객이 dependents가 있는지 여부(Yes, No).

            고객의 계정 정보
                tenure - 고객이 자사 서비스를 사용한 개월 수.
                Contract - 고객의 계약 기간 (Month-to-month, One year, Two year)
                PaperlessBilling - 고객이 paperless billing를 사용하는지 여부 (Yes, No)
                PaymentMethod - 고객의 지불 방법 (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
                MonthlyCharges - 고객에게 매월 청구되는 금액
                TotalCharges - 고객에게 총 청구된 금액

            고객이 가입한 서비스
                PhoneService - 고객이 전화 서비스를 사용하는지 여부(Yes, No).
                MultipleLines - 고객이 multiple line을 사용하는지 여부(Yes, No, No phone service).
                InternetService - 고객의 인터넷 서비스 사업자 (DSL, Fiber optic, No).
                OnlineSecurity - 고객이 online security 서비스를 사용하는지 여부 (Yes, No, No internet service)
                OnlineBackup - 고객이 online backup을 사용하는지 여부 (Yes, No, No internet service)
                DeviceProtection - 고객이 device protection에 가입했는지 여부 (Yes, No, No internet service)
                TechSupport 고객이 tech support를 받고있는지 여부 (Yes, No, No internet service)
                StreamingTV - 고객이 streaming TV 서비스를 사용하는지 여부 (Yes, No, No internet service)
                StreamingMovies - 고객이 streaming movies 서비스를 사용하는지 여부 (Yes, No, No internet service)

            문제 정의
                분석의 목적
                통신사의 고객 데이터에서 CLV를 계산합니다.
                통신사 고객의 churn 해지를 예측합니다.
    """

    print("\n", "=" * 5, "04", "=" * 5)
    df = pd.read_csv("./data_file/WA_Fn-UseC_-Telco-Customer-Churn.csv")


    # 데이터 확인
    print(df.shape)
    print(df.tail())

    # 결측값 측정
    print(df.info())
    # object - Yes Or No ( Category Data )
    # TotalCharges      7043 non-null   object
    print(df.isnull().sum())

    # df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    print(df[df['TotalCharges'] == ' '])
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    print(df[df['tenure'] == 0])
    df = df[df['TotalCharges'].notnull()]

    # float으로 변환
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    print(df.info(0))

    # 기술통계 확인 : df.describe()
    print(df.describe())


    # 변수간의 correlation 확인 df.corr() 시각화
    fig, axes = plt.subplots(2, 4, sharey=False, tight_layout=True, figsize=(15, 6), num='mobile game')

    corr = df.corr()
    print(corr)
    # annot=True 숫자 표시해줌

    ax_temp = axes[0, 0]
    ax_temp.set_title('HeatMap')
    sns.heatmap(corr, annot=True, ax=ax_temp)

    # 해지한 고객수 ( Count Flot )
    sns.countplot(y='Churn', data=df, ax=axes[0, 1])

    # 변수간의 pairplot
    """
        Pairplot으로 눈으로 볼 수 있는 관계는 이 정도 입니다.
        tenure가 낮은 경우 churn이 많습니다. 즉, 최근 고객들이 더 많이 해지합니다.
        어느정도 이상의 tenure이 되면 충성고객이 되어 churn하지 않는 것 같습니다.
        MonthlyCharges가 높은 경우의 churn이 많습니다.
        tenure과 MonthlyCharges가 아마도 주요한 변수인 것 같습니다.
        scatter plot을 봐도 어느 정도 경계선이 보입니다.
        pairplot으로 볼 수 있는 관계가 많지 않습니다.
        numeric variable이 많지 않기 때문입니다.
        categorical variable을 처리해줍니다.
    """
    sns.pairplot(data=df, hue='Churn', markers='+', palette='husl')

    # 카테고리 변수의 카테고리가 어떻게 구성되어 있는지 확인 합니다.
    print(df['gender'].value_counts())
    print("=================================")
    print(df['Partner'].value_counts())
    print("=================================")
    print(df['Dependents'].value_counts())
    print("=================================")
    print(df['PhoneService'].value_counts())
    print("=================================")
    print(df['MultipleLines'].value_counts())
    print("=================================")
    print(df['InternetService'].value_counts())
    print("=================================")
    print(df['OnlineSecurity'].value_counts())
    print("=================================")
    print(df['OnlineBackup'].value_counts())
    print("=================================")
    print(df['DeviceProtection'].value_counts())
    print("=================================")
    print(df['TechSupport'].value_counts())
    print("=================================")
    print(df['StreamingTV'].value_counts())
    print("=================================")
    print(df['StreamingMovies'].value_counts())
    print("=================================")
    print(df['Contract'].value_counts())
    print("=================================")
    print(df['PaperlessBilling'].value_counts())
    print("=================================")
    print(df['PaymentMethod'].value_counts())
    print("=================================")
    print(df['Churn'].value_counts())

    # 다음 컬럼들에 대해 'No internet service'를 'No'로 변환해줍니다.
    replace_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies']
    for i in replace_cols:
        df[i] = df[i].replace({'No internet service': 'No'})



    def barplot_percentages(feature, axes, orient='v', axis_name="percentage of customers"):
        ratios = pd.DataFrame()
        g = df.groupby(feature)["Churn"].value_counts().to_frame()
        g = g.rename({"Churn": axis_name}, axis=1).reset_index()
        g[axis_name] = g[axis_name] / len(df)
        if orient == 'v':
            sns.barplot(x=feature, y=axis_name, hue='Churn', data=g, orient=orient, ax=axes)
            axes.set_yticklabels(['{:,.0%}'.format(y) for y in axes.get_yticks()])
        else:
            sns.barplot(x=axis_name, y=feature, hue='Churn', data=g, orient=orient , ax=axes)
            axes.set_xticklabels(['{:,.0%}'.format(x) for x in axes.get_xticks()])
        axes.plot()

    # https://www.kaggle.com/jsaguiar/exploratory-analysis-with-seaborn

    # "SeniorCitizen"
    # SeniotCitizen은 전체 고객의 16% 정도에 불과하지만 churn 비율은 훨씬 높습니다. (42% vs 23%)
    fig, axes = plt.subplots(2, 4, sharey=False, tight_layout=True, figsize=(15, 6), num='data bar plot')
    barplot_percentages('SeniorCitizen', axes[0, 0])

    # 'Dependents'
    # Dependent가 없는 경우 churn을 더 많이 합니다.
    barplot_percentages('Dependents', axes[0, 1])

    # 'Partner'
    # Partner가 없는 경우 churn을 더 많이 합니다
    barplot_percentages('Partner', axes[0, 2])

    # "MultipleLines"
    # phone service를 사용하지 않는 고객의 비율은 적습니다.
    # MultipleLines를 사용중인 고객의 churn 비율이 아주 약간 높습니다.
    barplot_percentages('MultipleLines', axes[0, 3])

    # "InternetService"
    # 인터넷서비스가 없는 경우의 churn 비율은 매우 낮습니다.
    # Fiber opptic을 사용중인 고객이 DSL 사용중인 고객들보다 churn 비율이 높습니다.
    barplot_percentages('InternetService', axes[1, 0])

    # 6개의 부가 서비스관련 시각화 해봅니다.
    # "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport" 부가서비스 사용자는 churn 하는 경우가 적습니다.
    # 스트리밍 서비스 이용 고객 중 churn이 많은 것으로 보입니다. ("StreamingTV", "StreamingMovies")
    cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection"]
    df1 = pd.melt(df[df["InternetService"] != "No"][cols]).rename({'value': 'Has service'}, axis=1)
    sns.countplot(data=df1, x='variable', hue='Has service', ax=axes[1, 1])
    axes[1, 1].set(xlabel='Additional service', ylabel='Num of customers')

    cols = ["TechSupport", "StreamingTV", "StreamingMovies"]
    df1 = pd.melt(df[df["InternetService"] != "No"][cols]).rename({'value': 'Has service'}, axis=1)
    sns.countplot(data=df1, x='variable', hue='Has service', ax=axes[1, 2])
    axes[1, 1].set(xlabel='Additional service', ylabel='Num of customers')
    plt.show()

    # Contract 유형에 따른 월청구요금과 해지여부를 시각화 합니다.
    # 장기계약이고 월청구요금이 높을수록 해지율이 높은 것 같습니다.
    # 전반적으로 월청구요금이 높을때 해지가능성이 높아보입니다.
    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='data bar plot')

    sns.boxplot(data=df, x='Contract', y='MonthlyCharges', hue='Churn', ax=axes[0, 0])
    axes[0, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # PaymentMethod 유형에 따른 월청구요금과 해지여부를 시각화 합니다.
    # Mailed check는 상대적으로 월청구요금이 낮습니다.
    # Mailed check에서 해지고객과 비해지 고객의 차이가 큽니다.

    sns.boxplot(data=df, x='PaymentMethod', y='MonthlyCharges', hue='Churn', ax=axes[0, 1])
    axes[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # tenure에 따른 고객수를 계산합니다.
    print(df['tenure'].value_counts().sort_index())
    a = df['tenure'].value_counts().sort_index()
    print(a.shape)

    # tenure에 따른 고객수를 시각화
    # 6개월 이후 retention이 상당히 낮아진다는 것을 알 수 있습니다.
    # 반면, 장기 충성고객들은 70개월 이상 유지되고 있습니다. 소중한 고객들입니다.
    ax_temp = axes[1, 0]
    ax_temp.plot(np.arange(1, 73), a, 'o')
    ax_temp.plot(np.arange(1, 73), a, '-', alpha=0.8)
    ax_temp.set_xlabel('tenure')
    ax_temp.set_ylabel('Number of customer')

    plt.show()

    print("\n", "=" * 3, "01.", "=" * 3)

    # CLV 계산 및 활용방안
    # CAC와 함께 봐야하는 LTV

    """
        CLV(Customer Lifetime Value; LTV)를 계산합니다.
            CLV는 고객생애 가치를 이야기 합니다.
            고객이 확보된 이후 유지되는 기간동안의 가치입니다.
            CAC와 LTV는 반드시 트래킹해야할 주요 지표라고 할 수 있습니다.
            CAC보다 LTV가 최소 3배 이상 높은 것이 이상적입니다.
            LTV (Lifetime value)
            
        LTV (Lifetime value)    
            고객당 월 평균 이익(Avg monthly revenue per customer) x 평균 고객 유지개월 수(# months customer lifetime)
            고객당 월 평균 이익(Avg monthly revenue per customer) / 월 평균 해지율(Monthly churn)
            (Average Value of a Sale) x (Number of Repeat Transactions) x (Average Retention Time in Months or Years for a Typical Customer)
            PLC(제품수명주기; Product Life Cycle) x ARPU(고객평균매출; Average Revenue Per User)
            고객당 월 평균 이익(Avg Monthly Revenue per Customer x 고객당 매출 총 이익 (Gross Margin per Customer) / 월평균 해지율 (Monthly Churn Rate)
            
        CAC (Customer Acquisition Cost)
            전체 세일즈 마케팅 비용 (Total sales and marketing exppense) / # 신규확보 고객 수 (# New customers acquired)
            LTC:CAC Ratio
    
        LTV/CAC
            1:1 더 많이 팔수록 더 많이 잃게 됩니다.
            3:1 이상적인 비율입니다. (도메인마다 다를 수 있습니다.)
            4:1 좋은 비즈니스 모델입니다.
            5:1 마케팅에 투자를 덜 하고 있는 것으로 보입니다.
    """

    # * LTV (Lifetime value)
    #  - 고객당 월 평균 이익(Avg monthly revenue per customer) x 평균 고객 유지개월 수(# months customer lifetime)
    # LTV는 2100달러입니다.
    # CAC는 700달러 정도인 것이 이상적입니다.
    # 통신사의 CAC는 기기 보조금, 멤버십 혜택 등이 있습니다.

    ltv = df['MonthlyCharges'].mean() * df['tenure'].mean()
    print('LTV : ', ltv)
    print("\n", "=" * 3, "02.", "=" * 3)

    # Churn 해지할 고객을 예측합니다.
    df2 = df.iloc[:, 1:]
    print(df2.tail())

    # target Value Churn Yes / No -> 1 / 0
    # binary 형태의 카테고리 변수를 numeric variable로 변경해줍니다.
    df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
    df2['Churn'].replace(to_replace='No', value=0, inplace=True)

    # 모든 categorical 변수를 더미 변수화 시킵니다.
    df_dummies = pd.get_dummies(df2)
    print(df_dummies.shape, df_dummies.tail())

    # dummy 변수화한 데이터를 사용합니다.
    y = df_dummies['Churn'].values
    X = df_dummies.drop(columns='Churn')

    # 변수 값을 0과 1사이 값으로 스케일링 해줍니다.
    from sklearn.preprocessing import MinMaxScaler
    features = X.columns.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X))
    X.columns = features
    print(X.shape)
    print(X.tail())

    # Create Train & Test Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # logistic regression
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    result = model.fit(X_train, y_train)

    from sklearn import metrics
    prediction_test = model.predict(X_test)
    # Print the prediction accuracy
    print(metrics.accuracy_score(y_test, prediction_test))

    print("\n", "=" * 3, "03.", "=" * 3)

    # 모든 변수의 weights 값을 가져와서 시각화 합니다.
    weights = pd.Series(model.coef_[0], index=X.columns.values)
    plt.rcParams['figure.figsize'] = (20, 4)
    weights.sort_values(ascending=False).plot(kind='bar')

    """
        결과 해석 및 적용 방안
            데이터 탐색과정에서 주요한 변수일 것으로 보였던 변수들의 weight가 실제로 높습니다.
            trnure가 길수록 충성고객의 churn은 낮아집니다. tenure가 아주 긴 유저들의 churn이 낮은 것이 이렇게 나타난 것으로 보입니다. (TotalCharges는 반대로 같은이치)
            인터넷 서비스를 사용하지 않는 것은 고객의 churn을 줄입니다.
            Fiber optic 인터넷 서비스 사용과 월단위 계약, Electronic Check를 사용하는 고객일수록 churn이 높아집니다.
    """
    # RandomForest
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    model_rf = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1,
                                      random_state=50, max_features="auto",
                                      max_leaf_nodes=30)
    model_rf.fit(X_train, y_train)

    # Make predictions
    prediction_test = model_rf.predict(X_test)
    print(metrics.accuracy_score(y_test, prediction_test))

    importances = model_rf.feature_importances_
    weights = pd.Series(importances, index=X.columns.values)
    plt.rcParams["figure.figsize"] = (14, 4)
    weights.sort_values(ascending=False).plot(kind='bar')

    # random forest 알고리즘에서 monthly contract, tenure and total charges가 churn을 예측하는 가장 주요한 변수입니다.
    # logistic regression의 결과와 EDA 결과와 매우 유사합니다.

    """
    적용 방안
        중요도가 높은 변수를 활용한 마케팅 전략을 수립해봅니다.
        계약 조건을 변경해볼 수 있습니다. 2년 장기계약을 최대한 유도해봅니다.
        폰 보조금을 많이 지급해서 CAC가 높아지더라도 장기적으로 유지하여 LTV를 높인다면 통신사에게 더 유리합니다.
        Fiber opptic 을 사용할수록 해지확률이 높아지는데, 그 이유를 찾아봅니다. 인터넷 통신 통합요금제 등의 영향일 수 있습니다.
        etc
        매달 고객별 churn을 예측하여 churn할 것으로 예측되는 고객들을 대상으로 선행적 조치를 취합니다.
        예: 새 기기로 교체해주고 보조금을 지급한 뒤 2년 계약하는 쪽으로 유도하는 마케팅 전화를 돌려봅니다.
    """

    from sklearn.svm import SVC
    model.svm = SVC(kernel='linear')
    model.svm.fit(X_train, y_train)
    preds = model.svm.predict(X_test)
    metrics.accuracy_score(y_test, preds)

    # Create the Confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, preds))

    # ADA Boost (AdaBoost Algorithm)
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier()
    # n_estimators = 50 (default value)
    # base_estimator = DecisionTreeClassifier (default value)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics.accuracy_score(y_test, preds)

    # XG Boost
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics.accuracy_score(y_test, preds)


# marketing_04()


def marketing_05():
    """
        subject
            Machine_Running
        topic
            ex4. marketing_data
        content
            05. Referral
        Describe
            리뷰 분석을 통하 소비자 조사
        sub Contents
            01. 분석할 데이터 파악, 경쟁사 고객 리뷰

            STEP1. 형태소 분석
            STEP2. 불용어 처리

        텍스트 마이닝을 위한 전처리
            KoNLP를 이용한 형태소 분석
            KoNLPy가 제공하는 형태소분석기 중 하나인 Kkma를 사용합니다.
            자세한 내용은 http://konlpy.org/ko/v0.4.3/morph/ 참조
        형태소 분석기
            한나눔 http://semanticweb.kaist.ac.kr/hannanum/index.html
            트위터 https://github.com/twitter/twitter-korean-text
            꼬꼬마 http://kkma.snu.ac.kr/documents/
    """

    print("\n", "=" * 5, "05", "=" * 5)
    from konlpy.tag import Hannanum
    from konlpy.tag import Twitter
    from konlpy.tag import Kkma
    hannanum = Hannanum()
    twitter = Twitter()
    kkma = Kkma()

    """
        꼬꼬마 형태소 분석기
            문장을 형태소 단위로 분리하고 품사를 태깅합니다
            품사태그는 일반명사(NNG), 고유명사(NNP), 동사(VV), 형용사(VA) 등이 있습니다
            http://kkma.snu.ac.kr/documents/index.jsp?doc=postag 형태소 리스트를 확인
    """
    print(kkma.sentences(u'아버지가 방에 들어가셨다. 아버지 가방에 들어가셨다. 아버지가 방 안에 있는 가방에 들어가셨다.'))
    print(kkma.pos(u'아버지가 방에 들어가셨다. 아버지 가방에 들어가셨다. 아버지가 방 안에 있는 가방에 들어가셨다.'))
    print(hannanum.pos(u'아버지가 방에 들어가셨다. 아버지 가방에 들어가셨다. 아버지가 방 안에 있는 가방에 들어가셨다.'))
    print(twitter.pos('아버지가 방에 들어가셨다. 아버지 가방에 들어가셨다. 아버지가 방 안에 있는 가방에 들어가셨다.'))

    print("\n", "=" * 3, "01.", "=" * 3)
    """
        텍스트 마이닝 분석 및 시각화
            센트룸 데이터를 먼저 분석해봅니다.
    """

    line_list = []
    f = open("data_file/centrum_review.txt", encoding="utf-8")
    for line in f:
        line = kkma.nouns(line)
        line_list.append(line)
    f.close()
    print("- 불러온 문서 :", len(line_list), "문장")

    word_frequency = {}
    noun_list = []
    # 불용어 리스트를 여기에 추가합니다.
    stop_list = ["배송", "만족", '구매', '감사']
    line_number = 0
    for line in line_list[:]:
        line_number += 1
        print(str(line_number) + "/" + str(len(line_list)), end="\r")
        noun = []
        for word in line:
            if word.split("/")[0] not in stop_list and len(word.split("/")[0]) > 1:
                noun.append(word.split("/")[0])
                if word not in word_frequency.keys():
                    word_frequency[word] = 1
                else:
                    word_frequency[word] += 1
        noun_list.extend(noun)

    # 단어별 출현빈도를 출력합니다.
    word_count = []
    for n, freq in word_frequency.items():
        word_count.append([n, freq])
    word_count.sort(key=lambda elem: elem[1], reverse=True)
    for n, freq in word_count[:10]:
        print(n + "\t" + str(freq))
    # 추출한 명사 리스트를 활용해 명사만으로 이루어진 문서를 생성합니다.
    noun_doc = ' '.join(noun_list)
    noun_doc = noun_doc.strip()


    """
        서체 다운로드
            시각화에서 서체 변경만으로도 완성도를 높일 수 있습니다.
            다음의 링크에서 나눔스퀘어 서체를 다운로드 받아주세요.
            참고: https://hangeul.naver.com/2017/nanum
    """
    # 워드클라우드 시각화
    # pip install wordcloud

    from wordcloud import WordCloud, ImageColorGenerator
    import matplotlib.pyplot as plt

    # 폰트업로드

    # 워드클라우드 파라미터 설정
    font_path = "font/NanumSquareB.otf"  # 폰트
    background_color = "white"  # 배경색
    margin = 3  # 모서리 여백 넓이
    min_font_size = 7  # 최소 글자 크기
    max_font_size = 150  # 최대 글자 크기
    width = 500  # 이미지 가로 크기
    height = 500  # 이미지 세로 크기
    wc = WordCloud(font_path=font_path, background_color=background_color, margin=margin, \
                   min_font_size=min_font_size, max_font_size=max_font_size, width=width, height=height)
    wc.generate(noun_doc)

    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    """
        텍스트에서 토픽/주제 찾기
        LDA 토픽 모델링
    """

    # pip install gensim
    # pip install corpora
    # pip install wheel
    import gensim
    from gensim import corpora
    import logging
    logging.basicConfig(level=logging.DEBUG)
    topic = 5
    keyword = 10
    texts = []
    resultList = []
    stop_list = ["배송", "만족", "카페", "카페규정", "확인", "주수", "센트"]
    for line in line_list:
        words = line
        if words != [""]:
            tokens = [word for word in words if (len(word.split("/")[0]) > 1 and word.split("/")[0] not in stop_list)]
            texts.append(tokens)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic, id2word=dictionary, passes=10)
    for num in range(topic):
        resultList.append(ldamodel.show_topic(num, keyword))
    print("\n", "=" * 3, "03.", "=" * 3)

    print(resultList)
    # unsupervide running 이라서 분석할때마다 결과가 좀 달라짐.
    # gephi 시각화도 해보면 좋음..


marketing_05()


def marketing_temp():
    """
        subject
            Machine_Running
        topic
            ex4. marketing_data
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

# marketing_temp()

