"""
    subject
        Machine_Running
    topic
        예제로 학습하기
        포켓몬 데이터셋 ( 분류 분석 )

    Describe


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

def poketmon_01():
    """
        subject
            Machine_Running
        topic
            예제로 학습하기
            포켓몬 데이터셋 ( 분류 분석 )
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
        type_list = []
        type_list.append(x1)
        if x2 is not np.nan:
            type_list.append(x2)
        return type_list

    prog_df['Type'] = prog_df.apply(lambda x: make_list(x['Type 1'], x['Type 2']), axis=1)
    print(prog_df.head())

    del prog_df['Type 1']
    del prog_df['Type 2']
    from sklearn.preprocessing import MultiLabelBinarizer

    # 멀티 레이블 바이너리화 ( 2가지의 속성을 원핫 인코딩 한 값이라고 보면됨.)
    mlb = MultiLabelBinarizer()
    prog_df = prog_df.join(pd.DataFrame(mlb.fit_transform(prog_df.pop('Type')), columns=mlb.classes_))
    print(prog_df.head())

    prog_df = pd.get_dummies(prog_df, columns=['Generation'])
    print(prog_df.head())

    print("\n", "=" * 3, "03.", "=" * 3)
    # 피처 표준화
    from sklearn.preprocessing import StandardScaler
    sc_scaler = StandardScaler()
    scale_colums =['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    prog_df[scale_colums] = sc_scaler.fit_transform(prog_df[scale_colums])
    print(prog_df.head())

    x = prog_df.loc[:, prog_df.columns != 'Legendary']
    y = prog_df['Legendary']

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, random_state=30)
    print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # train LR
    lr = LogisticRegression(random_state=0)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_valid)

    print(accuracy_score(y_valid, y_pred))
    print(precision_score(y_valid, y_pred))
    print(recall_score(y_valid, y_pred))
    print(f1_score(y_valid, y_pred))
    from sklearn.metrics import confusion_matrix

    conf_mn = confusion_matrix(y_true=y_valid, y_pred=y_pred)
    print(conf_mn)

    print(prog_df['Legendary'].value_counts())

    # 클래스 불균형 해소를 위한 샘플링 변경
    positive_random_idx = prog_df[prog_df['Legendary'] == 1].sample(65, random_state=30).index.tolist()
    negative_random_idx = prog_df[prog_df['Legendary'] == 0].sample(65, random_state=30).index.tolist()

    random_idx = positive_random_idx + negative_random_idx

    x = prog_df.loc[random_idx, prog_df.columns != 'Legendary']
    y = prog_df['Legendary'][random_idx]
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, random_state=30)
    print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

    lr = LogisticRegression(random_state=0)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_valid)

    print(accuracy_score(y_valid, y_pred))
    print(precision_score(y_valid, y_pred))
    print(recall_score(y_valid, y_pred))
    print(f1_score(y_valid, y_pred))

    conf_mn = confusion_matrix(y_true=y_valid, y_pred=y_pred)
    print(conf_mn)

    # STEP 3 군집 분류 분석
    # 비지도 학습과 군집 분류 분석
    # K-means를 활용한 군집 분류
    # 주어진 데이터를 k개의 클러스터로 묶는 방식, 거리 차이의 분산을 최소화 하는 방식으로 동작
    # Expectation / Maximization
    # 차원수가 많을경우엔 차원 축소를 한후 사용하거나 다른 알고리즘을 사용한다.
    # 군집 분류 결과 해석과 시각화


def poketmon_03():
    """
        subject
            Machine_Running
        topic
            data analysis techniques
        content
            비지도 학습 기반 군집 분류 분석

            K-means를 활용한 군집 분류
                주어진 데이터를 k개의 클러스터로 묶는 방식, 거리 차이의 분산을 최소화 하는 방식으로 동작
                Expectation / Maximization
                차원수가 많을경우엔 차원 축소를 한후 사용하거나 다른 알고리즘을 사용한다.

            군집 분류 분석의 결과 해석
                차원 축소를 이용한 결과 해석
                엘보우 메서드 (elbow method)
                실루엣 계수 평가 등등..
        Describe

        sub Contents
            01.
    """
    print("\n", "=" * 5, "03", "=" * 5)
    from sklearn.cluster import KMeans
    df = pd.read_csv("./data_file/pokemon.csv")

    # 전처리
    df['Legendary'] = df['Legendary'].astype(int)
    df['Generation'] = df['Generation'].astype(int)
    prog_df = df[['Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation',
                  'Legendary']]

    # one-hot encoding
    encoded_df = pd.get_dummies(prog_df['Type 1'])
    print(encoded_df.head())

    def make_list(x1, x2):
        type_list = []
        type_list.append(x1)
        if x2 is not np.nan:
            type_list.append(x2)
        return type_list

    prog_df['Type'] = prog_df.apply(lambda x: make_list(x['Type 1'], x['Type 2']), axis=1)
    print(prog_df.head())

    del prog_df['Type 1']
    del prog_df['Type 2']
    from sklearn.preprocessing import MultiLabelBinarizer

    # 멀티 레이블 바이너리화 ( 2가지의 속성을 원핫 인코딩 한 값이라고 보면됨.)
    mlb = MultiLabelBinarizer()
    prog_df = prog_df.join(pd.DataFrame(mlb.fit_transform(prog_df.pop('Type')), columns=mlb.classes_))
    print(prog_df.head())

    prog_df = pd.get_dummies(prog_df, columns=['Generation'])
    print(prog_df.head())
    print("\n", "=" * 3, "01.", "=" * 3)

    x = prog_df[['Attack', 'Defense']]
    k_list = []
    cost_list = []
    # 2차원 군집 분석
    for k in range(1, 6):
        kmeans = KMeans(n_clusters=k).fit(x)
        inertia = kmeans.inertia_
        print('k:', k, '| cost:', inertia)
        k_list.append(k)
        cost_list.append(inertia)

    plt.plot(k_list, cost_list)
    plt.show()
    kmeans = KMeans(n_clusters=4).fit(x)
    cluster_num = kmeans.predict(x)
    cluster = pd.Series(cluster_num)
    prog_df['cluster_num'] = cluster.values
    print(prog_df.head(), prog_df['cluster_num'].value_counts())

    plt.scatter(prog_df[prog_df['cluster_num'] == 0]['Attack'],
                prog_df[prog_df['cluster_num'] == 0]['Defense'],
                s=50, c='red', label='Poketmon Group 1'
    )
    plt.scatter(prog_df[prog_df['cluster_num'] == 1]['Attack'],
                prog_df[prog_df['cluster_num'] == 1]['Defense'],
                s=50, c='blue', label='Poketmon Group 3'
    )
    plt.scatter(prog_df[prog_df['cluster_num'] == 2]['Attack'],
                prog_df[prog_df['cluster_num'] == 2]['Defense'],
                s=50, c='green', label='Poketmon Group 3'
    )
    plt.scatter(prog_df[prog_df['cluster_num'] == 3]['Attack'],
                prog_df[prog_df['cluster_num'] == 3]['Defense'],
                s=50, c='yellow', label='Poketmon Group 3'
    )
    plt.title('poketmon_cluster')
    plt.xlabel('Attack')
    plt.ylabel('Defense')
    plt.legend()
    plt.show()




    print("\n", "=" * 3, "02. 다차원 군집 분석", "=" * 3)

    x = prog_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']]
    k_list = []
    cost_list = []

    # 2차원 군집 분석
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k).fit(x)
        inertia = kmeans.inertia_
        print('k:', k, '| cost:', inertia)
        k_list.append(k)
        cost_list.append(inertia)

    plt.plot(k_list, cost_list)
    plt.show()
    kmeans = KMeans(n_clusters=6).fit(x)
    cluster_num = kmeans.predict(x)
    cluster = pd.Series(cluster_num)
    prog_df['cluster_num'] = cluster.values
    print(prog_df.head(), prog_df['cluster_num'].value_counts())

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='view')

    ax_temp = axes[0, 0]
    ax_temp.set_title('HP')
    sns.boxplot(x=cluster_num, y='HP', data=prog_df, ax=ax_temp)

    ax_temp = axes[0, 1]
    ax_temp.set_title('Attack')
    sns.boxplot(x=cluster_num, y='Attack', data=prog_df, ax=ax_temp)

    ax_temp = axes[1, 0]
    ax_temp.set_title('Defense')
    sns.boxplot(x=cluster_num, y='Defense', data=prog_df, ax=ax_temp)

    ax_temp = axes[1, 1]
    ax_temp.set_title('Speed')
    sns.boxplot(x=cluster_num, y='Speed', data=prog_df, ax=ax_temp)

    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)

poketmon_03()

def poketmon_temp():
    """
        subject
            Machine_Running
        topic
            예제로 학습하기
            포켓몬 데이터셋 ( 분류 분석 )
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


