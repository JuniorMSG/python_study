"""
    subject
        Machine_Running
    topic
        예제로 학습하기

    Describe
        Boston 부동산 데이터셋 ( EDA & 회귀 분석 )
        포켓몬 데이터셋 ( 분류 분석 )
        영화 본 슈프리머시 시나리오 파일 ( 텍스트 마이닝 )
        트립어드바이저 '제주 호텔' 리뷰 데이터 ( 감성 분류 )

        데이터 분석이 포함하는 내용은 다양하다.
        분석의 목적, 분야에 따라 여러가지의 기술들이 필요하다.

        BA(Business Analytics)
        Data Analytics
        Machine Learning Engineer
        Data Engineer
        Data Scientist
        Research Scientist

        모든 데이터 분석의 공통점
        1. 목표에 대한 문제 정의
        2. 문제 해결에 필요한 탐색적 데이터 분석
        3. 목표에 맞는 분석 기법 적용
            - 회귀 분석
            - 딥 러닝
            - 수학 기법 적용
            - 데이터 시각화 등등

        탐색적 데이터분석
        예측 분석
        분류 분석석
    Contens
        01. 차원축소
        02. 정밀도, 재현률, f1 score
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



def eda_01():
    """
        subject
            Machine_Running
        topic
            data analysis techniques
        content
            01. EDA (Exploratory Data Analysis) 탐색적 데이터 분석
        Describe
                        지도학습 - 머신러닝, 데이터 분석과 밀접한 관련이 있음.
            회귀분석, 요인분석, 예측분석, 분류분석에서 주로 사용함.

            회귀분석은 지도학습에 속하는 한가지 방석
            설명변수와 종속변수간의 인과관계를 찾아냄는 것.

            함수를 데이터에 맞추는 과정 ( 모델 학습 과정)
            OLS (Ordinary Least Square)많이 사용함.
            MLE (Maximum Likelihood Estimator)

            OLS (Ordinary Least Square) : 제곱을 가장 작은 상태로 추정하는것 - 오차들의 제곱을 최소화 하는 것
            그래디언트 디센트 (Gradient Decent)

        sub Contents
            01.
    """

    data = load_boston()
    df_boston = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_boston['MEDV'] = data['target']
    print(df_boston.head())
    print(df_boston.columns)
    # x_train, x_valid, y_train, y_valid = train_test_split(df_boston.drop('MEDV', 1), df_boston['MEDV'])
    numerical_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                         'LSTAT']
    cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    # TOWN : 지역 이름
    # LON, LAT : 위도, 경도 정보
    # MEDV : 해당 지역의 집값(중간값)
    # CRIM : 근방 범죄율
    # ZN : 주택지 비율
    # INDUS : 상업적 비즈니스에 활용되지 않는 농지 면적
    # CHAS : 경계선에 강에 있는지 여부
    # NOX : 산화 질소 농도
    # RM : 자택당 평균 방 갯수
    # AGE : 1940 년 이전에 건설된 비율
    # DIS : 5 개의 보스턴 고용 센터와의 거리에 다른 가중치 부여
    # RAD : radial 고속도로와의 접근성 지수
    # TAX : 10000달러당 재산세
    # PTRATIO : 지역별 학생-교사 비율
    # B : 지역의 흑인 지수 (1000(B - 0.63)^2), B는 흑인의 비율.
    # LSTAT : 빈곤층의 비율

    # shape 확인
    print(df_boston.shape)

    # 결측치 확인
    print(df_boston.isnull().sum())

    # 데이터 확인
    print(df_boston.info())



    print("\n", "=" * 5, "01", "=" * 5)
    print("\n", "=" * 3, "01. MEDV 피처 탐색", "=" * 3)

    # 회귀 분석 종속(목표) 변수 탐색
    # 중간값 확인
    print(df_boston['MEDV'].describe())

    # df_boston['MEDV'].hist(bins=50)
    # plt.show()

    # df_boston.boxplot(column=['MEDV'])
    # plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)

    # 회귀 분석 설명 변수 탐색
    # 목표 변수의 가격에 영향을 미치는 변수들에 대한 탐색

    fig = plt.figure(figsize=(17, 17))
    ax = fig.gca()
    df_boston[numerical_columns].hist(ax=ax)

    # 상관관계 탐색하기 피어슨 상관계수
    corr = df_boston[cols].corr(method='pearson')
    print(corr)
    fig = plt.figure(figsize=(17, 17))
    ax = fig.gca()
    sns.set(font_scale=1.5)
    hm = sns.heatmap(corr.values, annot=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols, ax=ax)
    plt.tight_layout()
    # 상관관계는 1에 가까워지면 양의 상관관계 -1에 가까워지면 음의 상관관계를 가진다.

    # RM 방의 개수
    plt.plot('RM', 'MEDV', data=df_boston, linestyle='none', marker='o', markersize=5, color='blue', alpha=0.5)
    plt.title('RM Scatter plot')
    plt.xlabel('RM')
    plt.xlabel('MEDV')

    # LSTAT 빈곤층의 비율
    plt.plot('LSTAT', 'MEDV', data=df_boston, linestyle='none', marker='o', markersize=5, color='blue', alpha=0.5)
    plt.title('LSTAT Scatter plot')
    plt.xlabel('LSTAT')
    plt.xlabel('MEDV')



    from sklearn.preprocessing import StandardScaler

    # 피처 표준화  feature standardization
    scaler = StandardScaler()
    df_boston[cols] = scaler.fit_transform(df_boston[cols])
    print(df_boston.head())

    # 데이터셋 분리
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(df_boston[cols], df_boston['MEDV'], test_size=0.2, random_state=30)
    plt.show()
    # 회귀 분석 모델 학습
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    lr = LinearRegression()
    lr_model = lr.fit(x_train, y_train)
    print(lr.coef_)

    plt.rcParams['figure.figsize'] = [12, 16]
    coefs = lr.coef_.tolist()
    coefs_series = pd.Series(coefs)
    x_labels = cols
    ax = coefs_series.plot.barh()
    ax.set_title('feature coef graph')
    ax.set_xlabel('coef')
    ax.set_ylabel('x_featires')
    ax.set_yticklabels(x_labels)

    plt.show()
    # R2 score - RMSE score 계산
    print('R2 score : lr_model.score(x_train, y_train) :', lr_model.score(x_train, y_train))
    print('R2 score : lr_model.score(x_test, y_test)   :', lr_model.score(x_test, y_test))

    mse_train =  mean_squared_error(y_train, lr.predict(x_train))
    mse_test = mean_squared_error(y_test, lr.predict(x_test))
    print('MSE : ', 'mse_train :', mse_train)
    print('MSE : ', 'mse_test :', mse_test)
    print('RMSE : ', 'rmse_train', sqrt(mse_train))
    print('RMSE : ', 'rmse_test', sqrt(mse_test))

    print(mean_squared_error(y_test, lr.predict(x_test)))

    # 피처 유의성 검정
    import statsmodels.api as sm
    x_train = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # 다중 공선성? vir 계수
    # 10이상일 경우 다른 feature들과 상관관계가 높다.
    vif = pd.DataFrame()
    vif['VIF Facotr'] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]
    vif['feature'] = x_train.columns
    print(vif)


print("\n", "=" * 3, "03.", "=" * 3)

# eda_01()


def eda_02():
    """
        subject
            Machine_Running
        topic
            data analysis techniques
        content
            02. 분류분석 - 포캣몬 데이터셋
        Describe
            탐색적 데이터 분석
                - 데이터셋 기초 정보 탐색
                - 개별 피처의 정보 탐색
                - 그룹 단위 특성 탐색
            이진 분류 분석
                - 로지스틱 분류 모델 (Logistic Regreession)에 대한 이해
                - 분류 분석에 필요한 전처리 기법
                - 분류 모델의 결과 해석과 평가
            군집 분류 분석
                - 비지도 학습과 군집 분류 분석
                - K-means를 활용한 군집 분류
                - 군집 분류 결과 해석과 시각화

            포켓몬 능력치, 타입, 세대, 종류 에 따른 분류
            DataSet -
                Name : 포켓몬 이름
                Type 1 : 포켓몬 타입 1
                Type 2 : 포켓몬 타입 2
                Total : 포켓몬 총 능력치 (Sum of Attack, Sp. Atk, Defense, Sp. Def, Speed and HP)
                HP : 포켓몬 HP 능력치
                Attack : 포켓몬 Attack 능력치
                Defense : 포켓몬 Defense 능력치
                Sp. Atk : 포켓몬 Sp. Atk 능력치
                Sp. Def : 포켓몬 Sp. Def 능력치
                Speed : 포켓몬 Speed 능력치
                Generation : 포켓몬 세대
                Legendary : 전설의 포켓몬 여부

        ub Contents
            01.
    """
    print("\n", "=" * 5, "02", "=" * 5)
    df = pd.read_csv("./data_file/pokemon.csv")
    print(df.head())
    print('shape', df.shape)
    print('결측치', df.isnull().sum())

    print('전설', df['Legendary'].value_counts())
    print('세대', df['Generation'].value_counts())
    df['Generation'].value_counts().sort_index().plot()
    plt.show()

    print('type1 :', df['Type 1'].unique())
    print('type2 :', df['Type 2'].unique())

    print("\n", "=" * 3, "01.", "=" * 3)

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='view')
    axes[0, 0].set_title('All Box')
    sns.boxplot(ax=axes[0, 0], data=df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']])

    axes[0, 1].set_title('Legendary Box')
    sns.boxplot(ax=axes[0, 1], data=df[df['Legendary'] == 1][['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']])
    plt.show()

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='Legendary Group')
    axes[0, 0].set_title('Type 1')
    df['Type 1'].value_counts(sort=False).sort_index().plot.barh(ax=axes[0, 0])

    axes[0, 1].set_title('Legendary Type 1')
    df[df['Legendary'] == 1]['Type 1'].value_counts(sort=False).sort_index().plot.barh(ax=axes[0, 1])

    axes[1, 0].set_title('Type 2')
    df['Type 2'].value_counts(sort=False).sort_index().plot.barh(ax=axes[1, 0])

    axes[1, 1].set_title('Legendary Type 2')
    df[df['Legendary'] == 1]['Type 2'].value_counts(sort=False).sort_index().plot.barh(ax=axes[1, 1])

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='Generation Group')

    axes[0, 0].set_title('Generation')
    df['Generation'].value_counts(sort=False).sort_index().plot.barh(ax=axes[0, 0])

    axes[0, 1].set_title('Legendary Generation ')
    df[df['Legendary'] == 1]['Generation'].value_counts(sort=False).sort_index().plot.barh(ax=axes[0, 1])

    axes[1, 0].set_title('Legendary Generation ')
    groups = df[df['Legendary'] == 1].groupby('Generation').size()
    groups.plot.bar(ax=axes[1, 0])

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='Generation Total')

    ax_temp = axes[0, 0]
    ax_temp.set_title('Generation Total')
    sns.boxplot(x='Generation', y='Total', data=df, ax=ax_temp)

    ax_temp = axes[0, 1]
    ax_temp.set_title('Type 1 Total')
    sns.boxplot(x='Generation', y='Total', hue='Legendary', data=df, ax=ax_temp)

    ax_temp = axes[1, 0]
    ax_temp.set_title('Type 1 Total')
    sns.boxplot(x='Type 1', y='Total', data=df, ax=ax_temp)

    ax_temp = axes[1, 1]
    ax_temp.set_title('Type 2 Total')
    sns.boxplot(x='Type 2', y='Total', data=df, ax=ax_temp)

    # plt.show()

    print("\n", "=" * 3, "02.", "=" * 3)
    # 분류분석, 로지스틱 회귀 모델이란?
    # 이진 분류 (Binary Classification), 다중 분류 (Multi-Class Classification),
    # 이진 분류와 다중 분류는 class가 미리 정해져있음.
    # 군집 분류 ( Clustering) 군집 분류는 클래스가 미리 정해져있지 않음.

    # 전처리
    df['Legendary'] = df['Legendary'].astype(int)
    df['Generation'] = df['Generation'].astype(int)
    prog_df = df[['Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']]

    # one-hot encoding
    encoded_df = pd.get_dummies(prog_df['Type 1'])
    print(encoded_df.head())

    def make_list(x1, x2):
        list = []
        list.append(x1)
        if x2 is not np.nan:
            list.append(x2)
        return list
    x1, x2 = prog_df['Type 1'], prog_df['Type 2']
    prog_df['Type'] = prog_df['Type 1'].apply(lambda x: make_list([x1, x2]), axis=1)
    print(prog_df.head())


    print("\n", "=" * 3, "03.", "=" * 3)


eda_02()


def eda_temp():
    """
        subject
            Machine_Running
        topic
            data analysis techniques
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

# eda_temp()


