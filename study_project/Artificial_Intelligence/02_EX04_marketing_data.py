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

    # 분석에 필요한 컬럼만 선택
    # df = df[['userid', 'version', 'sum_gamerounds', 'retention_1', 'retention_7']]

    # 변수간의 correlation 확인 df.corr() 시각화
    corr = df.corr()
    print(corr)

    ax_temp = axes[0, 0]
    ax_temp.set_title('HeatMap')
    sns.heatmap(corr, annot=True, ax=ax_temp)

    # ax_temp = axes[0, 1]
    # ax_temp.set_title('sum_gamerounds box plot')
    # sns.boxenplot(data=df, y='', ax=ax_temp)

    plt.show()



    print("\n", "=" * 5, "03", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


marketing_03()

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

