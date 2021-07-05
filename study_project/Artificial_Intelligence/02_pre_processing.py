"""
    subject
        Machine_Running
    topic
        Preprocessing & data
    Describe
        머신러닝 학습용 데이터 전처리
    Contents
        01. 데이터 용어 설명
        02. 전처리 용어 설명

"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import os

def preprocessing_01():
    """
        subject
            Machine_Running
        topic
            Preprocessing

        content
            01. 데이터 용어 설명

                학습 데이터, 예측 데이터,
                (features, labels, train, test)

                검증 데이터
                    - 과대적합 (OVERFITTING)        : 너무 과하게 분류할경우 실데이터 적용시 성능이 떨어짐
                    - 과소적합 (UNDERFITTING)       : 데이터 적어서 정상적으로 분류하지 못함.
                    - OPTIMUM                      : 적합한 지점

        Describe
            데이터 용어 설명

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
    """
    print("\n", "=" * 5, "03", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# preprocessing_01()


def preprocessing_02():
    """
        subject
            Machine_Running
        topic
            Preprocessing

        content
            02. 전처리 용어 설명
        Describe
            01. 전처리 (pre-processing)
                - 데이터 분석에 적합하게 데이터를 가공/변형/처리/클리닝
                Garbage in, Garbage out!
                데이터 수집 및 전처리에 80%정도의 시간을 사용한다고 한다.
                1. 결측치 : 데이터의 빠진 부분을 어떻게 채워줄지?
                2. 이상치 : 데이터가 이상할때 어떻게 처리할지?
                3. 정규화 (Normalization)          : 0~1사이의 분포로 조정
                4. 표준화 (Standardization)        : 평균을 0, 표준편차를 1로 맞춤
                5. 샘플링 (over/under sampling)    : 적은 데이터 / 많은 데이터를 줄이거나 늘려서 샘플 개수를 맞추는 작업
                6. 피처 공학 (Feature Engineering)
                    - feature 생성 / 연산
                    - 구간 생성 (20대, 30대등), 스케일 변형
        sub Contents
            01.
    """
    print("\n", "=" * 5, "02", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# preprocessing_02()


def preprocessing_03():
    """
        subject
            Machine_Running
        topic
            Preprocessing
        content
            03. 데이터 전처리 해보기.
        Describe

        sub Contents
            01. train / validation 세트 나누기
                1. feature, label을 정의하고
                2. feature / label을 정의했으면 적절한 비율로 train / valdation set을 나눕니다.
            02. 결측치
                https://scikit-learn.org/stable/modules/impute.html
    """
    print("\n", "=" * 5, "03. 데이터 전처리 해보기", "=" * 5)
    titanic = sns.load_dataset('titanic')
    print(titanic.head(), titanic.columns)
    print("\n", "=" * 3, "01.", "=" * 3)
    feature = ['pclass', 'sex', 'age', 'fare']
    label = ['survived']

    # 학습 데이터
    x_train_tatinic = titanic[feature]
    # 예측 데이터
    y_train_tatinic = titanic[label]
    print(x_train_tatinic.head(), y_train_tatinic)

    # shuffle=True(디폴드 True) : 나눠주면서 섞을것인가?
    # test_size (validation data의 비율)
    # random_state : 섞을때 동일하게 섞기 위해서 사용한다...
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_tatinic, y_train_tatinic, test_size=0.2, shuffle=True, random_state=30)
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)

    print("\n", "=" * 3, "02. 숫자형 데이터 결측치 처리 ", "=" * 3)
    print(titanic.info())

    # 결측치 개수 출력
    print(titanic.isnull().sum())
    print(titanic['age'].isnull().sum())

    # 숫자형 데이터
    print(titanic['age'].fillna(0).describe())
    print(titanic['age'].fillna(titanic['age'].mean()).describe())

    from sklearn.impute import SimpleImputer
    # 순서대로 하는방법 mean - 평균
    titanic = sns.load_dataset('titanic')
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(titanic[['age', 'pclass']])
    result = imputer.transform(titanic[['age', 'pclass']])
    titanic[['age', 'pclass']] = result
    print(titanic[['age', 'pclass']].isnull().sum())
    print(titanic[['age', 'pclass']].describe())
    print(result)

    # 한번에 하는방법 mean - 평균
    titanic = sns.load_dataset('titanic')
    imputer = SimpleImputer(strategy='median')
    titanic[['age', 'pclass']] = imputer.fit_transform(titanic[['age', 'pclass']])
    print(titanic[['age', 'pclass']].isnull().sum())
    print(titanic[['age', 'pclass']].describe())

    print("\n", "=" * 3, "03. 문자 (Categorical Column) 데이터에 대한 결측치 처리", "=" * 3)

    # 한번에 하는방법 mean - 평균
    titanic = sns.load_dataset('titanic')
    titanic['embarked'].fillna('S')

    # most_frequent 가장 빈도수가 높은 값으로 채운다.
    imputer = SimpleImputer(strategy='most_frequent')
    titanic[['embarked', 'deck']] = imputer.fit_transform(titanic[['embarked', 'deck']])
    print(titanic[['embarked', 'deck']] .isnull().sum())


# preprocessing_03()


def preprocessing_04():
    """
        subject
            Machine_Running
        topic
            preprocessing_Running

        content
            04. Label Encoding : 문자(Categorical)를 수치(numberical)로 변환
        Describe
            데이터 학습을 위해서 모든 문자로된 데이터는 수치로 변환하여야 한다.
        sub Contents
            01.
    """
    print("\n", "=" * 5, "04", "=" * 5)

    print("\n", "=" * 3, "01. 함수 방법", "=" * 3)
    titanic = sns.load_dataset('titanic')
    # def
    def convert(data):
        if data == 'male':
            return 1
        elif data == 'female':
            return 0

    print(titanic['sex'].value_counts())
    titanic['sex'] = titanic['sex'].apply(convert)
    print(titanic['sex'].value_counts())

    print("\n", "=" * 3, "02. sklearn 이용", "=" * 3)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer

    titanic = sns.load_dataset('titanic')
    le = LabelEncoder()

    print(titanic['sex'].value_counts())
    titanic['sex'] = le.fit_transform((titanic['sex']))
    print(titanic['sex'].value_counts())
    titanic['sex'] = le.inverse_transform(titanic['sex'])
    print(titanic['sex'].value_counts())

    print(titanic.isnull().sum())
    imputer = SimpleImputer(strategy='most_frequent')
    # 결측치 채우기
    titanic[['embarked', 'deck']] = imputer.fit_transform(titanic[['embarked', 'deck']])
    print(titanic[['embarked', 'deck']] .isnull().sum())

    # 2개씩은 안됨..
    titanic['embarked'] = le.fit_transform(titanic['embarked'])
    titanic['deck'] = le.fit_transform(titanic['deck'])
    print(titanic.isnull().sum())
    print(titanic['deck'].value_counts())
    print(titanic['embarked'].value_counts())

    print("\n", "=" * 3, "03. ", "=" * 3)


# preprocessing_04()


def preprocessing_05():
    """
        subject
            Machine_Running
        topic
            preprocessing_Running
        content
            05. one hot encoding
        Describe
            column을 분리시켜 카테고리형 -> 수치형 변환에서 생기는 수치형 값의 관계를 끊어주어서 독립적인 형태로 바꿔준다.

            원 핫 인코딩이란
            원핫 인코딩은 카테고리(계절, 성별, 종류등)의 특성을 가지는 column에 대해서 적용 해야 한다.

            데이터를 기계학습 시킬때, S='2', Q='1'이면 Q+Q=S가 된다라고 학습하게 된다.
            독립적인 데이터는 별도의 column으로 분리하고,
            각각의 컬럼에 해당 값에만 True 나머지는 False를 가지도록 하는것이다.

        sub Contents
            01. 예시
    """
    print("\n", "=" * 5, "05", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    titanic = sns.load_dataset('titanic')
    print(titanic['embarked'].value_counts())
    print(titanic['embarked'].isnull().sum())

    titanic['embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(titanic[['embarked']])
    print(titanic['embarked'].value_counts())
    print(titanic['embarked'].isnull().sum())

    titanic['embarked_num'] = LabelEncoder().fit_transform((titanic['embarked']))
    print(titanic['embarked_num'].value_counts())

    one_hot = pd.get_dummies(titanic['embarked_num'][:6])
    one_hot.columns = ['C', 'Q', 'S']
    print(one_hot)


    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


# preprocessing_05()


def preprocessing_06():
    """
        subject
            Machine_Running
        topic
            preprocessing_Running

        content
            06. 정규화 (Normalize), 표준화 (Standard Scaling)
        Describe
            scale(규모)을 맞추는 작업

            정규화 (Normalize)
            column 간에 다른 min, max 값을 가지는 경우, 정규화를 통해 최소치, 최대값의 척도를 맞추어 주는것을 말한다.

            표준화 (Standard Scaling)
            평균이 0과 표준편차가 1이 되도록 변환함

        sub Contents
            01. 정규화 (Normalize) 예시
            02. 표준화 (Standard Scaling) 예시
    """
    print("\n", "=" * 5, "06. 정규화 (Normalize)", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    Grades = {
        'A_School': [1, 2, 2.5, 3.3, 3.7],
        'B_School': [0.5, 1.5, 2.5, 3.5, 4.5],
        'C_School': [20, 40, 60, 80, 100]
    }
    Grades = pd.DataFrame(data=Grades)
    print(Grades)

    from sklearn.preprocessing import MinMaxScaler

    min_max_grades = MinMaxScaler().fit_transform(Grades)
    print(pd.DataFrame(min_max_grades, columns=['A_School', 'B_School', 'C_School']))


    print("\n", "=" * 3, "02.", "=" * 3)
    from sklearn.preprocessing import StandardScaler

    standard_scaler = StandardScaler()

    x = np.arange(10)
    x[5] = 500
    x[9] = 1000
    print('mean :', x.mean(), 'std :', x.std())
    scaled = standard_scaler.fit_transform((x.reshape(-1, 1)))
    print('mean :', scaled.mean(), 'std :', scaled.std())
    print(round(scaled.mean()), round(scaled.std()))


# preprocessing_06()


def preprocessing_07():
    """
        subject
            Machine_Running
        topic
            preprocessing_Running

        content
            07. sklearn.dataset에서 제공해주는 샘플 데이터 활용하기

        Describe
            숫자형 데이터셋 : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html?highlight=load_digit#sklearn.datasets.load_digits
            아이리스 데이터셋 : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html?highlight=load_iris#sklearn.datasets.load_iris
                DESCR : 데이터셋의 정보를 보여준다
                data : feature data
                feature_names : feature data의 컬럼 이름
                target : label data (수치형)
                target_names : label의 이름 (문자형)
        sub Contents
            01.
    """

    import warnings

    print("\n", "=" * 5, "07. sklearn.dataset에서 제공해주는 샘플 데이터 활용하기", "=" * 5)
    # iris data set

    from sklearn.datasets import load_iris

    iris = load_iris()
    # DESCR : 데이터셋의 정보를 보여준다
    print(iris['DESCR'])
    # data : feature data
    # feature_names : feature data의 컬럼 이름
    # target : label data (수치형)
    # target_names : label의 이름 (문자형)

    data = iris['data'][:5]
    print(data)
    data = iris['feature_names'][:5]
    print(data)

    # sepal : 꼿 받침
    # peatal : 꽃잎


    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

preprocessing_07()

def preprocessing_temp():
    """
        subject
            Machine_Running
        topic
            preprocessing_Running

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

# preprocessing_temp()