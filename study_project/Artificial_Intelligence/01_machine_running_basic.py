"""
    subject
        Machine_Running
    topic
        머신 러닝의 개요
    Describe
        01. 머신러닝의 개요
            인공지능    : 사람의 지능을 모방하여, 사람이 하는 것과 같이 복잡한 일을 할 수 있게 기계를 만드는 것
            머신러닝    : 기본적으로 알고리즘을 이용해 데이터를 분석 및 학습하며, 학습한 내용을 기반으로 판단, 예측함
            딥러닝      : 인공신경망에서 발전한 형태의 인공 지능, 머신러닝 중 하나의 방법론

            데이터(Data)를 기반으로 패턴(Model - 알고리즘)을 학습하여 결과를 추론(Prediction)하는 것
            머신러닝 - 패턴(Model - 알고리즘)을 스스로 학습한다.

            지도학습, 비지도학습, 강화학습

            지도학습 (Supervised Learning) : 데이터 O - 모델 - 결과값 O
                회귀 (Regression)     : 수치형(numeric value) 집 값, 가격, 온도, 기온등등
                분류 (Classifcation)  : 종류 판별 : 개/고양이, 스팸매일: 스팸/정상 등등

            비지도학습 (Unsupervised Learning) : 데이터 O - 모델 - 결과값 X
                군집화 (Clustering)                     : 뉴스 분류, 사용자 관심사
                차원 축소 (Dimentionality Reduction)    :

            머신러닝 장점
                1. 복잡한 해턴을 인지할 수 있다.
                2. 적절한 알고리즘, 다양한 양질의 데이터가 있다면, 좋은 성능
                3. 도메인 영역(업무 영역)에 대한 지식이 상대적으로 부족해도 가능하다.
            머신러닝 단점
                1. 데이터의 의존성이 크다 (Garbage in, Garbage out)
                2. 과적합의 오류에 빠질 수 있다 (일반화 오류 - 데이터 편향성, 데이터 다양성 요구)
                3. 풍부한 데이터가 기본적으로 요구된다.

            좋은 성능 = 좋은 데이터 (quality, quantity)
            데이터 가공 - 전처리 (pre-processing)

            데이터, 예측해야 할 값에 맞는 알고리즘을 적용 해야한다.

        02. 가설 함수(Hypothesis), 비용(Cost), 손실 함수(Lostt Function),
            가설 함수(Hypothesis)
                - x = 1, 2, 3
                - y = 3, 5, 7
                H(x) = W * X + b


    Contents
        pip install scikit-learn
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        # 모델선언
            model = LinearRegression()
        # 학습
            model.fit(x,y)
        # 예측
            prediction = model.predict(x2)


        scikit-learn
            classification
            Regression
            Clustering
            Dimensionality reduction
            Model selection
            Preprocessing
        https://scikit-learn.org

"""


import numpy as np
from sklearn.linear_model import LinearRegression


def machine_02():
    """
        subject
            Machine_Running
        topic
            Machine_Running

        content
            02. sklearn 간단하게 머신러닝 실행해보기
        Describe

        sub Contents
            01.
    """
    print("\n", "=" * 5, "02. sklearn 간단하게 머신러닝 실행해보기", "=" * 5)

    print("\n", "=" * 3, "01.", "=" * 3)
    # reshape x, y, z 차원으로 바꾼 배열을 리턴함.
    x = np.arange(10).reshape(-1, 1)
    y = (3*x + 1).reshape(-1, 1)

    print(x, y)

    # 모델 선언
    model = LinearRegression()
    print(model)

    # model.fit(데이터, 예측값) 학습시키기
    model.fit(x, y)

    prediction_01 = model.predict([[10]])
    prediction_02 = model.predict(np.arange(95).reshape(-1, 1))
    print(prediction_01)
    print(prediction_02)

    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


machine_02()


def machine_03():
    """
        subject
            Machine_Running
        topic
            Machine_Running

        content
            03. 용어설명
                학습 데이터, 예측 데이터,
                (features, labels, train, test)

                검증 데이터
                    - 과대적합 (OVERFITTING)        : 너무 과하게 분류할경우 실데이터 적용시 성능이 떨어짐
                    - 과소적합 (UNDERFITTING)       : 데이터 적어서 정상적으로 분류하지 못함.
                    - OPTIMUM                      : 적합한 지점

        Describe
            용어 설명

        sub Contents
            01. 학습 데이터, 예측 데이터

                model = LinearRegression()
                model.fit(데이터 - features, 예측값 - labels)

                x_train, y_train : 학습을 위한 데이터 (Traning Set)
                    - 모델이 학습하기 위해 필요한 데이터
                    - feature/label 둘다 존재

                x_test  : 예측을 위한 데이터 (Test_Set)
                    - 모델이 예측하기 위한 데이터
                    - feature만 존재
                    - y_test를 예측해야 한다.

                3단계 구성
                - 모델생성
                    model = LinearRegression()
                - 학습
                    model.fit(x_train, y_train)
                - 예측
                    prediction = model.predict(x_test)

            02. 검증 데이터 (Validation Set)
                Traning Set 8 : 2 Validation Set
                Train data로 학습하고 validation data로 모니터링

            03. 전처리 (pre-processing)
                - 데이터 분석에 적합하게 데이터를 가공/변형/처리/클리닝
                Garbage in, Garbage out!
                데이터 수집 및 전처리에 80%정도의 시간을 사용한다고 한다.
                1. 결측치 (Imputer) : 데이터의 빠진 부분을 어떻게 채워줄지?
                2. 이상치 : 데이터가 이상할때 어떻게 처리할지?
                3. 정규화 (Normalization)          : 0~1사이의 분포로 조정
                4. 표준화 (Standardization)        : 평균을 0, 표준편차를 1로 맞춤
                5. 샘플링 (over/under sampling)    : 적은 데이터 / 많은 데이터를 줄이거나 늘려서 샘플 개수를 맞추는 작업
                6. 피처 공학 (Feature Engineering)
                    - feature 생성 / 연산
                    - 구간 생성 (20대, 30대등), 스케일 변형

    """
    print("\n", "=" * 5, "03", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

machine_03()

def machine_temp():
    """
        subject
            Machine_Running
        topic
            Machine_Running

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

# machine_temp()