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

data = load_boston()
df_boston = pd.DataFrame(data['data'], columns=data['feature_names'])
df_boston['MEDV'] = data['target']
print(df_boston.head())
print(df_boston.columns)
# x_train, x_valid, y_train, y_valid = train_test_split(df_boston.drop('MEDV', 1), df_boston['MEDV'])
numerical_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
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


def eda_01():
    """
        subject
            Machine_Running
        topic
            data analysis techniques
        content
            01. EDA (Exploratory Data Analysis) 탐색적 데이터 분석
        Describe

        sub Contents
            01.
    """
    global df_boston, cols , numerical_columns
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



print("\n", "=" * 3, "03.", "=" * 3)

eda_01()


def eda_02():
    """
        subject
            Machine_Running
        topic
            data analysis techniques
        content
            02. 지도학습과 회귀분석
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

        ub Contents
            01.
    """


    print("\n", "=" * 5, "01", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# eda_01()


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


