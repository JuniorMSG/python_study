"""
    Chapter 01. 딥러닝 소개
        Step 01. 딥러닝이란?
            딥러닝의 등장, 인공지능/머신러닝/딥러닝의 관계에 대해 간단히 소개

            DL은 인공지능이라는 거대한 학문의 일부
            DL은 ML의 여러 기법 중 하나

        Step 02. 딥러닝의 원리
            기존 프로그래밍 방식과 다른 딥러닝의 학습 방식을 소개합니다.

            ML의 정의
                주어진 데이터나 과거 용례를 통하여 문제에 대한 해결 성능을 최대화하는 것
            Role of Statistics
                Inference from samples
            Role of Computer Science
                최적화 문제 해결
                추론을 위한 모델을 구성하고 평가


            프로그래밍의 경우
                규칙을 코딩한다.
            딥러닝의 경우
                모델을 디자인한다.


        Step 03. 딥러닝이 주로 다루는 문제들
            딥러닝으로 다룰 수 있는 문제군들을 소개합니다.

            Association Rule Mining

            Supervised Learning
                Classification
                Regression

                학습셋이 주어짐
                학습셋은 정답이 Labelled 되어있음
                주어진 학습셋을 근거로 학습
                새로운 데이터가 들어오면 정답을 예측


            Unsupervised Learning
                Clustering
                Feature Extraction
                Dimensionality Reduction

                데이터가 주어짐
                학습셋은 정답이 Labelled 되어있음
                주어진 데이터의 정보를 활용, 패턴을 추출하거나 관계를 구성
                명확하게 정해진 정답은 없음음

            Rinforcement Learning
                Unsupervised Learning의 일종
                그러나 정해진 데이터에 의해 학습되어지는 것은 아님
                보상/벌점의 개념을 모델링
                그 후 모델의 행위에 맞게 일어나는 경험을 학습





    Chapter 02. 딥러닝의 원리
        Step 01. 딥러닝의 구조
            집값 모델을 기준으로 딥러닝의 예측에 대하여 설명합니다.

            노가다의 끝
            언제 계산을 멈출 수 있나요?
                무한한 h(x)중 무엇이 좋은 함수 인가?
                찾는다면 어떻게 찾을 수 있는가?
            Idea
                나쁨/틀림(wrongness)를 정의
                Wrongness를 minimize 시키는 전략
                Cost/Loss function을 통한

                h(x)의 예측 y와 주어진 학습셋의 y의 차이를 최소화화

        Step 0. 비용 함수
            학습시 모델의 좋고 나쁨을 어떻게 평가하는지에 대하여 설명합니다.

        Step 03. 경사하강법 (GDM)
            학습시 어떻게 모델이 결과를 개선시키는지에 대하여 설명합니다.

    Chapter 03. 딥러닝 공부법
        Step 01. 딥러닝 라이브러리
            딥러닝 공부를 시작하며 알아야 할 대표 프레임웍들을 소개합니다.

        Step 02. 딥러닝을 제대로 공부하는 법
            결국, '문제해결' 역량이 중점 요소임을 강조하며 문제를 풀어나가며 학습하는 방식을 소개합니다.
"""