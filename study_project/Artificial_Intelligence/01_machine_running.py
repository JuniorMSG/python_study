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