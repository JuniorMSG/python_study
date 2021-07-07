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

import matplotlib.pyplot as plt
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

# preprocessing_04(


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
            preprocessing classification

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

    print("\n", "=" * 3, "01.", "=" * 3)
    # iris data set

    from sklearn.datasets import load_iris

    iris = load_iris()
    # DESCR : 데이터셋의 정보를 보여준다
    print(iris['DESCR'])
    # data : feature data
    # feature_names : feature data의 컬럼 이름
    # target : label data (수치형)
    # target_names : label의 이름 (문자형)

    data = iris['data']
    print(data[:5])
    feature_names = iris['feature_names']
    print(feature_names[:5])

    target = iris['target']
    print(target[:5])

    # sepal : 꼿 받침
    # peatal : 꽃잎
    print("\n", "=" * 3, "02.", "=" * 3)
    df_iris = pd.DataFrame(data, columns=feature_names)
    df_iris['target'] = target
    print(df_iris.head())

    sns.scatterplot('sepal length (cm)', 'sepal width (cm)', hue='target', palette='spring', data=df_iris)
    plt.title('sepal')
    plt.show()

    sns.scatterplot('petal width (cm)', 'petal length (cm)', hue='target', palette='spring', data=df_iris)
    plt.title('petal')
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)
    # 차원 축소
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA

    fig = plt.figure(figsize=(9, 6), num='Demension reduce')
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduce = PCA(n_components=3).fit_transform(df_iris.drop('target', 1))
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1], X_reduce[:, 2], c=df_iris['target'], cmap=plt.cm.Set1, edgecolors='k', s=40)
    ax.set_title('iris 3D')
    ax.set_xlabel('x')
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel('y')
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel('z')
    ax.w_zaxis.set_ticklabels([])

    plt.show()

    print("\n", "=" * 3, "04. 머신러닝 데이터셋 만들기", "=" * 3)
    from sklearn.model_selection import train_test_split

    # drop 메서드를 이용하면 선택한 값이 삭제된 새로운 객체를 얻을 수 있음.
    # train_test_split(feature data, target data)

    x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'])

    print('train : ', x_train.shape, y_train.shape)
    print('valid : ', x_valid.shape, y_valid.shape)

    sns.countplot(y_train)
    plt.show()

    # stratify 특정 칼럼 기준으로 클래스의 분포를 균등하게 배분한다.
    x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'], stratify=df_iris['target'])
    print('train : ', x_train.shape, y_train.shape)
    print('valid : ', x_valid.shape, y_valid.shape)

    sns.countplot(y_train)
    plt.show()

# preprocessing_07()


def preprocessing_08():
    """
        subject
            Machine_Running
        topic
            preprocessing classification

        content
            08. Logistic Regression (로지스틱 회귀)
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        Describe
            로지스틱 회귀(Logistic Regression)는 영국의 통계학자인 D. R. Cox가 1958년에 제안한 확률 모델
            독립 변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법

            Logistic Regression, 서포트 백터 머신 (SVM)과 같은 알고리즘은 이진 부류만 가능한데 (2개의 클래스만)
            3개 이상의 클래스에 대한 판별을 진행하는 경우 특정한 전략이 필요합니다.

            OvR 전략을 선호한다.
            one-vs-rest(OvR)    : K개의 클래스가 존재할 때, 1개의 클래스를 제외한 다른 클래스를 K개 만들어,
                                  각각의 이진 분류에 대한 확률을 구하고 총합을 통해 최종 클래스를 판별

            one-vs-one(OvO)     : 4개의 계절을 구분하는 클래스가 존재한다고 가정했을 때
                                  0vs1, 0vs2, 0vs3, 1vs2, 1vs3, 2vs3 까지 Nx(n-1)/2 개의 분류기를 만들어서
                                  가장 많이 양성으로 선택된 클래스를 판별  4*3 / 2 = 6

        sub Contents
            01.
    """
    print("\n", "=" * 5, "08. Logistic Regression (로지스틱 회귀)", "=" * 5)

    print("\n", "=" * 3, "01.", "=" * 3)
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    # 데이터 선언
    iris = load_iris()
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df_iris['target'] = iris['target']
    x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'], stratify=df_iris['target'])

    # 모델 선언
    ir_model = LogisticRegression(max_iter=500)
    # 모델 학습
    ir_model.fit(x_train, y_train)
    # 예측
    ir_pred = ir_model.predict(x_valid)
    print(ir_pred[:10])
    # 평가
    print((ir_pred == y_valid).mean())


# preprocessing_08()


def preprocessing_09():
    """
        subject
            Machine_Running
        topic
            preprocessing classification

        content
            09. stochastic gradient descent (SGD) : 확률적 경사 하강법
            https://scikit-learn.org/stable/modules/linear_model.html
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        Describe
            오차가 가장 적은 지점을 찾아 나갈때
        sub Contents
            01.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets import load_iris
    # 데이터 선언
    iris = load_iris()
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df_iris['target'] = iris['target']
    x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'], stratify=df_iris['target'])

    print("\n", "=" * 5, "09. stochastic gradient descent (SGD) : 확률적 경사 하강법", "=" * 5)

    print("\n", "=" * 3, "01.", "=" * 3)
    # 모델 선언
    sgd = SGDClassifier()

    # 모델 학습
    sgd.fit(x_train, y_train)

    # 예측
    prediction = sgd.predict(x_valid)
    print((prediction == y_valid).mean())

    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


# preprocessing_09()


def preprocessing_10():
    """
        subject
            Machine_Running
        topic
            preprocessing classification

        content
            10. 하이퍼 파라미터 (hyper-parameter) 튜닝
        Describe
            모델 선언시 전달하는 파라미터 옵션을 말한다.
            알고리즘마다 다르기 때문에 Document를 참조한다.
        sub Contents
            01.
    """
    print("\n", "=" * 5, "10", "=" * 5)
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets import load_iris
    # 데이터 선언
    iris = load_iris()
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df_iris['target'] = iris['target']
    x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'], stratify=df_iris['target'])

    print("\n", "=" * 3, "01.", "=" * 3)
    # random_statue=0 하이퍼 파라미터 값을 튜닝할 때는 random_state를 고정해야 한다.
    # n_jobs 멀티코어 ( -1은 전체 활용 )
    sgd = SGDClassifier(penalty='elasticnet', random_state=0, n_jobs=-1)
    sgd.fit(x_train, y_train)
    prediction = sgd.predict(x_valid)
    print((prediction == y_valid).mean())

    sgd = SGDClassifier(penalty='l1', random_state=0, n_jobs=-1)
    sgd.fit(x_train, y_train)
    prediction = sgd.predict(x_valid)
    print((prediction == y_valid).mean())

    sgd = SGDClassifier(random_state=0)
    sgd.fit(x_train, y_train)
    prediction = sgd.predict(x_valid)
    print((prediction == y_valid).mean())

    print("\n", "=" * 3, "02.", "=" * 3)


# preprocessing_10()


def preprocessing_11():
    """
        subject
            Machine_Running
        topic
            preprocessing classification

        content
            11. KNeighborsClassifier (최근접 이웃 알고리즘)
        Describe
            K-최근접 이웃 (K-Nearest Neighbors) 알고리즘은 분류(Classifier)와 회귀(Regression)에 모두 쓰입니다.
            처음 접하는 사람들도 이해하기 쉬운 알고리즘이며, 단순한 데이터를 대상으로 분류나 회귀를 할 때 사용합니다.

            복잡한 데이터셋에는 K-Nearest Neighbors 알고리즘은 제대로 된 성능발휘를 하기 힘듭니다.
            https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn

        sub Contents
            01.
    """
    print("\n", "=" * 5, "11", "=" * 5)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_iris
    # 데이터 선언
    iris = load_iris()
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df_iris['target'] = iris['target']
    x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'], stratify=df_iris['target'])

    print("\n", "=" * 3, "01.", "=" * 3)
    knc = KNeighborsClassifier()
    knc.fit(x_train, y_train)
    knc_pred = knc.predict(x_valid)
    print((knc_pred == y_valid).mean())

    knc = KNeighborsClassifier(n_neighbors=3)
    knc.fit(x_train, y_train)
    knc_pred = knc.predict(x_valid)
    print((knc_pred == y_valid).mean())

    knc = KNeighborsClassifier(n_neighbors=9)
    knc.fit(x_train, y_train)
    knc_pred = knc.predict(x_valid)
    print((knc_pred == y_valid).mean())


# preprocessing_11()


def preprocessing_12():
    """
        subject
            Machine_Running
        topic
            preprocessing classification

        content
            12. SVC ( 서포트 벡터 머신 )
            https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        Describe
            새로운 데이터가 어느 카테고리에 속할지 판단하는 비 확률적 이진 선형 분류 모델
            경계로 표현되는 데이터들 중 가장 큰 폭을 가진 경계를 찾는 알고리즘

            Logistic Regression (로지스틱 회귀)와 같이 이진 분류만 가능하다.
            OvsR 전략을 사용한다.

        sub Contents
            01.
    """
    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    # 데이터 선언
    iris = load_iris()
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df_iris['target'] = iris['target']
    x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'], stratify=df_iris['target'])

    print("\n", "=" * 5, "12", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    # 선언 - 학습 - 예측
    svc = SVC(random_state=0)
    svc.fit(x_train, y_train)
    svc_pred = svc.predict(x_valid)
    print((svc_pred == y_valid).mean())

    svc = SVC(random_state=1)
    svc.fit(x_train, y_train)
    svc_pred = svc.predict(x_valid)
    print((svc_pred == y_valid).mean())

    print(svc_pred[:5])
    # 클래스별 확률을 알려주는 decision_function
    print(svc.decision_function(x_valid)[:5])
    

    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


# preprocessing_12()


def preprocessing_13():
    """
        subject
            Machine_Running
        topic
            preprocessing classification

        content
            13. Decision Tree (의사 결정 나무)
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Describe
            스무고개처럼, 나무 가지치기를 통해 소그룹으로 나누어 판별하는 것
            알고리즘 성능이 나쁘지 않아서 많이 사용한다.
        sub Contents
            01. 간단 예시1
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 데이터 선언
    iris = load_iris()
    df_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df_iris['target'] = iris['target']
    x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', 1), df_iris['target'], stratify=df_iris['target'])

    print("\n", "=" * 5, "13. Decision Tree (의사 결정 나무)", "=" * 5)


    print("\n", "=" * 3, "01.", "=" * 3)
    dtc = DecisionTreeClassifier(random_state=0)
    dtc.fit(x_train, y_train)
    dtc_pred = dtc.predict(x_valid)
    print((dtc_pred == y_valid).mean())

    dtc = DecisionTreeClassifier(min_samples_split=5, random_state=0)
    dtc.fit(x_train, y_train)
    dtc_pred = dtc.predict(x_valid)
    print((dtc_pred == y_valid).mean())

    print("\n", "=" * 3, "02.", "=" * 3)

    # gini 계수 : 불순도를 의미하며, 계수가 높을 수록 엔트로피가 크다는 의미이다.
    # 엔트로피가 크다 => 클래스가 혼잡하게 섞여 있다.

    def graph_tree(model):
        from sklearn.tree import export_graphviz
        from subprocess import call

        import pydot


        path = os.path.dirname(os.path.abspath(__file__))

        # .dot 파일로 export 해줍니다
        export_graphviz(model, out_file='tree.dot')

        # dot 파일 읽어서 PNG 만들기
        # 실행이 제대로 안되서 conda에 다시 설치했음.
        # pip install graphviz
        # conda install graphviz
        # vscode에서 실행 안되는데 이유를 모르겠네
        (graph,) = pydot.graph_from_dot_file(path + '/tree.dot')
        graph.write_png('image/pre_processing/13.Decision_load_dot.png')

    graph_tree(dtc)

    from sklearn import tree
    # sklearn의 tree 사용해서 보내기
    fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    cn = ['setosa', 'versicolor', 'virginica']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    # tree.plot_tree(dtc, feature_names=fn, class_names=cn, filled=True)c
    tree.plot_tree(dtc)
    print(os.path.dirname(os.path.abspath(__file__)))

    fig.savefig('image/pre_processing/13.Decision_tree_01.png')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    # tree.plot_tree(dtc, feature_names=fn, class_names=cn, filled=True)
    tree.plot_tree(dtc, feature_names=fn, class_names=cn, filled=True)
    fig.savefig('image/pre_processing/13.Decision_tree_02.png')

    print("\n", "=" * 3, "03.", "=" * 3)
    # max_depthp가 너무 깊으면 과적합 오류가 발생할 수 있다.

    dtc = DecisionTreeClassifier(min_samples_split=5, random_state=0, max_depth=2)
    dtc.fit(x_train, y_train)
    dtc_pred = dtc.predict(x_valid)
    print((dtc_pred == y_valid).mean())

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    # tree.plot_tree(dtc, feature_names=fn, class_names=cn, filled=True)
    tree.plot_tree(dtc, feature_names=fn, class_names=cn, filled=True)
    fig.savefig('image/pre_processing/13.Decision_tree_max_depth_01.png')


# preprocessing_13()


def preprocessing_14():
    """
        subject
            Machine_Running
        topic
            preprocessing classification

        content
            14. 오차 구하기, 정확도란?
        Describe

            True Positive       : 실제 Positive인 정답을 Positive라고 예측 (True)
            Tree Negegative     : 실제 Negegative 정답을 Negegative 예측 (True)
            False Positive      : 실제 Negegative 정답을 Positive 예측 (False)     - 1종 오류
            False Negegative    : 실제 Positive   정답을 Negegative 예측 (False)   - 2종 오류

            1종 오류는 병이 걸린 사람을 병이 걸리지 않았다고 진단하는 경우이며.
            2종 오류는 병이 걸리지 않은 사람을 병이 걸렷다고 진단하는 경우이다.
            여기선 1종 오류가 더 치명적이며 상황에 따라 다르다.

            이러한 오류값을 처리하기 위해서 사용하는 방법이 있다.
            Accuracy (정확도)
            confusion matrix (오차 행렬)
            precision (정밀도)
            recall (재현율)

            정확도는 아래처럼 구해진다.
                 TP + TN
            ------------------
            TP + TN + FP + FN

            정확도의 역설은 실제 데이터에 Nagative 혹은 Positive 비율이 너무 높아서 희박한 가능성으로 발생할 상황에 대해
            제대로 된 분류를 평가 할 수 없다는 것.

            ex1) 데이터 수집을 긍정적인 지표만 수집하여 계산식을 작성할 경우
            ex2) 수집된 데이터중 긍정적인 지표만 사용할 경우


        sub Contents
            01. Accuracy (정확도)
            02. 정확도의 역설 (Accuracy Paradox) 예시

    """
    print("\n", "=" * 5, "14", "=" * 5)
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    import numpy as np
    cancer = load_breast_cancer()
    print(cancer['DESCR'])
    df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df_cancer['target'] = cancer['target']

    pos = df_cancer.loc[df_cancer['target'] == 1]
    neg = df_cancer.loc[df_cancer['target'] == 0]

    # 357개
    print('pos :', pos.value_counts())

    # 212개
    print('neg :', neg.value_counts())

    print("\n", "=" * 3, "01. 정상 데이터", "=" * 3)

    sample_general = pd.concat([pos, neg], sort=True)
    x_train, x_valid, y_train, y_valid = train_test_split(sample_general.drop('target', 1), sample_general['target'], random_state=42)
    model_general = LogisticRegression(max_iter=5000)
    model_general.fit(x_train, y_train)
    pred_general = model_general.predict(x_valid)

    print('일반적인 데이터', (pred_general == y_valid).mean())

    pred_error = np.ones(shape=y_valid.shape)
    print('데이터 강제 조정시', (pred_error == y_valid).mean())


    print("\n", "=" * 3, "02. 모수 줄인 데이터", "=" * 3)

    # 모수 줄인 데이터
    sample_divide = pd.concat([pos, neg[:5]], sort=True)
    x_train, x_valid, y_train, y_valid = train_test_split(sample_divide.drop('target', 1), sample_divide['target'], random_state=42)
    model_divide = LogisticRegression(max_iter=5000)
    model_divide.fit(x_train, y_train)
    pred_divide = model_divide.predict(x_valid)

    print('일반적인 데이터', (pred_divide == y_valid).mean())
    pred_error = np.ones(shape=y_valid.shape)
    print('데이터 강제 조정시', (pred_error == y_valid).mean())

    print("\n", "=" * 3, "03.", "=" * 3)


preprocessing_14()



def preprocessing_temp():
    """
        subject
            Machine_Running
        topic
            preprocessing classification

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