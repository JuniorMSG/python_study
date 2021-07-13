"""
    subject
        Machine_Running
    topic
        비지도학습 (Unsupervised Learning)

    Describe
        비지도학습 (Unsupervised Learning)은 기계 학습의 일종으로, 데이터가 어떻게 구성되었는지를 알아내는 문제의 범주
        지도 학습(Supervised Learning) 혹은 강화 학습(Reinforcement Learning)과는 달리 입력값에 대한 목표치가 주어지지 않는다.
        데이터는 있는데 y값이 없음.
        레이블 값이 없음.


        - 차원축소  : PCA, LDA, SVD
        - 군집화   : KMeans Clustering, DBSCAN
        - 군집화 평가
    Contents
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

print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility

range_n_clusters = [2, 3, 4, 5, 6]

def plot_silhouetee(X, num_cluesters):

    for n_clusters in num_cluesters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
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







def unsupervised_learning_01():
    """
        subject
            Machine_Running
        topic
            비지도학습 (Unsupervised Learning)
        content
            01. 차원축소
        Describe
            - feature의 개수를 줄이는 것, 특징을 추출하는 역활을 하기도함
            - 계산 비용을 감소시킴
            - 전반적인 데이터에 대한 이해도를 높임.

        sub Contents
            01. PCA(Principal component analysis) 차원축소
                - 주성분 분석 (PCA)는 선형 차원 축소 기법으로 매우 인기 있게 사용되는 차원 축소 기법중 하나
                - 주요 특징으로 분산(variance)을 최대한 보존한다는 점
                components
                    - 1보다 작은 값 - 분산을 기준으로 차원 축소
                    - 1보다 큰 값을 넣으면, 해당 값을 기준으로 feature를 축소

                # https://excelsior-cjh.tistory.com/167
            02. LDA(Linear Discriminant Analysis) 차원 축소
                - 선형 판별 분석법 (PCA와 유사)
                - LDA는 클래스(Class) 분리를 최대화하는 축을 찾기 위해 클래스 간 분산과 내부 분산의 비율을 최대화 하는 방식으로 차원 축소합니다.

            03. SVD (Singular Value Decomposition
                - 상품의 추천 시스템에도 활용되어지는 알고리즘 (추천시스템)
                - 특이값 분해기법이다.
                - PCA와 유사한 차원 축소 기법
                - scikit-learn 패키지에서는 truncated SVD (ala LSA)을 사용한다
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn import datasets

    print("\n", "=" * 5, "01", "=" * 5)

    iris = datasets.load_iris()
    data = iris['data']
    df_iris = pd.DataFrame(data, columns=iris['feature_names'])
    df_iris['target'] = iris['target']
    print(df_iris.head())


    print("\n", "=" * 3, "01.", "=" * 3)
    pca = PCA(n_components=2)
    data_scaled = StandardScaler().fit_transform(df_iris.loc[:, 'sepal length (cm)':'petal width (cm)'])
    pca_data = pca.fit_transform(data_scaled)
    print(pca_data[:5])

    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df_iris['target'])
    plt.show()


    pca = PCA(n_components=0.99)
    data_scaled = StandardScaler().fit_transform(df_iris.loc[:, 'sepal length (cm)':'petal width (cm)'])
    pca_data = pca.fit_transform(data_scaled)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    sample_size = 50
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], alpha=0.6, c=df_iris['target'])
    plt.savefig('./tmp.svg')
    plt.title('ax.plot')

    plt.show()


    print("\n", "=" * 3, "02.", "=" * 3)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    data_scaled = StandardScaler().fit_transform(df_iris.loc[:, 'sepal length (cm)':'petal width (cm)'])
    lda_data = lda.fit_transform(data_scaled, df_iris['target'])
    print(lda_data[:5])

    plt.scatter(lda_data[:, 0], lda_data[:, 1], c=df_iris['target'])
    plt.show()

    print("\n", "=" * 3, "03.", "=" * 3)

    from sklearn.decomposition import TruncatedSVD
    # https://ko.wikipedia.org/wiki/%ED%8A%B9%EC%9E%87%EA%B0%92_%EB%B6%84%ED%95%B4

    svd = TruncatedSVD(n_components=2)
    data_scaled = StandardScaler().fit_transform(df_iris.loc[:, 'sepal length (cm)':'petal width (cm)'])
    svd_data = svd.fit_transform(data_scaled, df_iris['target'])
    print(svd_data[:5])

    plt.scatter(svd_data[:, 0], svd_data[:, 1], c=df_iris['target'])
    plt.show()


# unsupervised_learning_01()


def unsupervised_learning_02():
    """
        subject
            Machine_Running
        topic
            비지도학습 (Unsupervised Learning)
        content
            02. 군집화 ( Clustering )
        Describe
            - 비지도학습의 대표적인 기술로 x에대한 레이블이 지정 되어있지 않은 데이터를 그룹핑하는 분석 알고리즘
            - 데이터들의 특성을 고려해 데이터 집단(클러스터)을 정의하고 데이터 집단의 대표할 수 있는 중심점을 찾는 것으로 데이터 마이닝의 한 방법이다.
            - 클러스터란 비슷한 특성을 가진 데이터들의 집단이다.
            - 데이터의 특성이 다르면 다른 클러스터에 속해야 한다.

        sub Contents
            01. K-Means Clustering
                군집화에서 가장 대중적으로 사용되는 알고리즘입니다. centroid라는 중점을 기준으로 가장 가까운 포인트들을 선택하는 군집화 기법입니다.
                # https://ko.wikipedia.org/wiki/K-%ED%8F%89%EA%B7%A0_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98

            02. DBSCAN (Density-based spatial clustering of applications with noise)
                밀도 기반 클러스터링
                - 밀도가 높은 부분을 클러스터링 하는 방식
                - 어느점을 기준으로 반경 x내에 점이 n개 이상 있으면 하나의 군집으로 인식하는 방식
                - KMeans 에서는 n_cluster의 갯수를 반드시 지정해 주어야 하나, DBSCAN에서는 필요없음
                - 기하학적인 clustering도 잘 찾아냄
    """

    from sklearn.cluster import KMeans
    from sklearn import datasets

    print("\n", "=" * 5, "02", "=" * 5)
    iris = datasets.load_iris()
    data = iris['data']
    df_iris = pd.DataFrame(data, columns=iris['feature_names'])
    df_iris['target'] = iris['target']
    print(df_iris.head())

    print("\n", "=" * 3, "01.", "=" * 3)
    # clusters 몇개의 집합이 있는지 설정하는 값 몇개인지 모를경우 여러번 해봐야함.
    kmeans = KMeans(n_clusters=3)
    kmeans_data = kmeans.fit_transform(df_iris.loc[:, 'sepal length (cm)':'petal width (cm)'])
    print(kmeans_data[:5])
    print(kmeans.labels_)

    fig, axes = plt.subplots(2, 2, tight_layout=True, figsize=(15, 6), num='view')
    fig.suptitle('view', fontsize=15)

    axes[0, 0].set_title('target')
    axes[0, 1].set_title('kmeans.labels_')
    sns.countplot(df_iris['target'], ax=axes[0, 0])
    sns.countplot(kmeans.labels_, ax=axes[0, 1])


    print("\n", "=" * 3, "02.", "=" * 3)
    kmeans = KMeans(n_clusters=3, max_iter=500)
    kmeans_data = kmeans.fit_transform(df_iris.loc[:, 'sepal length (cm)':'petal width (cm)'])
    print(kmeans_data[:5])
    print(kmeans.labels_)

    axes[1, 0].set_title('n_clusters=3, max_iter=500')
    sns.countplot(kmeans.labels_, ax=axes[1, 0])

    print("\n", "=" * 3, "03.", "=" * 3)

    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=0.3, min_samples=2)
    dbscan_data = dbscan.fit_predict(df_iris.loc[:, 'sepal length (cm)':'petal width (cm)'])
    print(dbscan_data)
    axes[1, 1].set_title('dbscan_data')
    sns.countplot(dbscan_data, ax=axes[1, 1])
    plt.show()

# unsupervised_learning_02()


def unsupervised_learning_03():
    """
        subject
            Machine_Running
        topic
            비지도학습 (Unsupervised Learning)
        content
            03. 군집 평가(Cluster Evaluation)
        Describe
            대부분의 군집화 데이터 셋은 비교할만한 데이터 셋을 가지고 있지 않다. 또한 군집화는 분류와 유사해 보일 수 있으나 성격이 많이 다르다.
            데이터 내에 숨어 있는 별도의 그룹을 찾아서 의미를 부여하거나 동일한 분류 값에 속하더라도
            그 안에서 더 세분화된 군집화를 추구하거나 서로 다른 분류 값의 데이터도 더 넓은 군집화 레벨하 등의 영역을 가지고 있다.
            비지도 학습의 특성상 어떠한 지표라도 정확하게 성능을 평가하기는 어렵다.
            그럼에도 불구하고 군집화의 성능을 평가하는 대표적인 방법으로 실루엣 분석을 이용한다.

        sub Contents
            01. 실루엣 스코어
                - 1 클러스터링의 품질이 좋다.
                - 0 클러스터링의 품질이 안좋다 ( 클러스터링이 의미가 없다 )
                - 음수 : 잘못 분류됨
    """
    print("\n", "=" * 5, "03", "=" * 5)
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.cluster import KMeans
    from sklearn import datasets

    print("\n", "=" * 5, "02", "=" * 5)
    iris = datasets.load_iris()
    data = iris['data']
    df_iris = pd.DataFrame(data, columns=iris['feature_names'])
    df_iris['target'] = iris['target']
    print(df_iris.head())


    print("\n", "=" * 3, "01.", "=" * 3)

    kmeans = KMeans(n_clusters=3)
    kmeans_data = kmeans.fit_transform(df_iris.loc[:, 'sepal length (cm)':'petal width (cm)'])
    print(kmeans_data[:5])
    print(kmeans.labels_)
    score = silhouette_score(kmeans_data , kmeans.labels_)
    print(score)

    print("\n", "=" * 3, "02.", "=" * 3)
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    plot_silhouetee(kmeans_data, [2,3,4,5,6,7,8,9,10])
    print("\n", "=" * 3, "03.", "=" * 3)


unsupervised_learning_03()


def unsupervised_learning_temp():
    """
        subject
            Machine_Running
        topic
            비지도학습 (Unsupervised Learning)
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

# unsupervised_learning_temp()


