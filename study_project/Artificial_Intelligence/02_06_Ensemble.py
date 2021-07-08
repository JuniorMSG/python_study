"""
    subject
        Machine_Running
    topic
        앙상블 (Ensemble) 예측

        https://scikit-learn.org/stable/
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    Describe
        머신러닝 앙상블이란 여러개의 머신러닝 모델을 이용해 최적의 답을 찾아내는 기법이다.
            - 여러 모델을 이용하여 데이터를 학습하고, 모든 모델의 예측 결과를 평균하여 예측한다.
            - 여러 개의 단일 모델들의 평균치를 내거나, 투표를 해서 다수결에 의한 결정을 하는 등 여러 모델들의 집단 지성을 활용하여 더 나은 결과를 도출해 내는 것에 주 목적이 있습니다.
        앙상블 기법의 종류
            - 보팅 (Voting) 투표를 통해 결과 도출
            - 배깅 (Bagging) 샘플 중복 생성을 통해 결과 도출
            - 부스팅 (Boosting) 이전 오차를 보완하면서 가중치 부여
            - 스태킹 (Stacking) 여러 모델을 기반으로 예측된 결과를 통해 meta 모델을 다시 예측

        여러 개의 단일 모델들의 평균치를 내거나, 투표를 해서 다수결에 의한 결정을 하는 등 여러 모델들의 집단 지성을 활용하여 더 나은 결과를 도출해 내는 것에 주 목적이 있습니다.
        https://teddylee777.github.io/machine-learning/ensemble%EA%B8%B0%EB%B2%95%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4%EC%99%80-%EC%A2%85%EB%A5%98-1
        https://teddylee777.github.io/machine-learning/ensemble%EA%B8%B0%EB%B2%95%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4%EC%99%80-%EC%A2%85%EB%A5%98-2
        https://teddylee777.github.io/machine-learning/ensemble%EA%B8%B0%EB%B2%95%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4%EC%99%80-%EC%A2%85%EB%A5%98-3

    Contents
        01. 보팅 (Voting) - 회귀 (Regression)
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


def font_set():
    from matplotlib import font_manager, rc
    font_path = "C:\Windows\Fonts\HYGTRE.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font)
    # RuntimeWarning: Glyph 8722 missing from current font.
    plt.rc('axes', unicode_minus=False)

my_predictions_mse = {}
my_predictions_mae = {}
colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
          ]


def plot_predictions(name_, pred, actual, axes):
    df = pd.DataFrame({'prediction': pred, 'actual': actual})
    df = df.sort_values(by='actual').reset_index(drop=True)

    sns.scatterplot(ax=axes, data=df['prediction'], marker='x', color='r')
    sns.scatterplot(ax=axes, data=df['actual'], alpha=0.7, marker='o', color='black')

    axes.set_title(name_, fontsize=15)
    axes.legend(['prediction', 'actual'], fontsize=12)


def set_data(name_, pred, actual):
    global my_predictions_mse
    global my_predictions_mae

    # MSE 값 측정
    mse = mean_squared_error(pred, actual)
    # MAE 값 측정
    mae = mean_absolute_error(pred, actual)

    my_predictions_mse[name_] = mse
    my_predictions_mae[name_] = mae

    y_value = sorted(my_predictions_mse.items(), key=lambda x: x[1], reverse=True)
    df_mse = pd.DataFrame(y_value, columns=['model', 'mse'])
    y_value = sorted(my_predictions_mae.items(), key=lambda x: x[1], reverse=True)
    df_mae = pd.DataFrame(y_value, columns=['model', 'mae'])

    return df_mse, df_mae


def view_graph(name_, pred, actual):
    global colors

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='view')
    fig.suptitle('view', fontsize=15)

    plot_predictions(name_, pred, actual, axes[0, 0])

    df_mse, df_mae = set_data(name_, pred, actual)
    min_ = df_mse['mse'].min() - 10
    max_ = df_mse['mse'].max() + 10

    axes[0, 1].set_yticks(np.arange(len(df_mse)))
    axes[0, 1].set_yticklabels(df_mse['model'], fontsize=15)
    axes[0, 1].set_title('MSE Error', fontsize=18)

    bars = axes[0, 1].barh(np.arange(len(df_mse)), df_mse['mse'])

    for i, v in enumerate(df_mse['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        axes[0, 1].text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    axes[0, 1].set_xlim(min_, max_)

    min_ = df_mae['mae'].min() - 10
    max_ = df_mae['mae'].max() + 10

    axes[1, 0].set_yticks(np.arange(len(df_mae)))
    axes[1, 0].set_yticklabels(df_mae['model'], fontsize=15)
    axes[1, 0].set_title('MAE ERROR', fontsize=18)

    bars = axes[1, 0].barh(np.arange(len(df_mae)), df_mae['mae'])

    for i, v in enumerate(df_mae['mae']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        axes[1, 0].text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    axes[1, 0].set_xlim(min_, max_)
    plt.show()


def compare_data_set():
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.pipeline import make_pipeline

    models = []

    # 비교군이 너무 많아서 가중치는 4개로 줄임.
    weights = [100, 1, 0.1, 0.001]
    data = load_boston()
    df_boston = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_boston['MEDV'] = data['target']
    x_train, x_valid, y_train, y_valid = train_test_split(df_boston.drop('MEDV', 1), df_boston['MEDV'])

    print("\n", "=" * 3, "01. 라쏘 (Lasso) - L1 규제를 활용한 모델", "=" * 3)
    for weight in weights:
        lasso = Lasso(alpha=weight, max_iter=1000)
        # 모델에 추가해둠
        models.append(('lasso_' + str(weight), lasso))

        lasso.fit(x_train, y_train)
        pred = lasso.predict(x_valid)
        set_data('Lasso(alpha={}'.format(weight), pred, y_valid)

    print("\n", "=" * 3, "02. 릿지 (Ridge) - L2 규제를 활용한 모델", "=" * 3)
    for weight in weights:
        ridge = Ridge(alpha=weight , max_iter=1000)
        # 모델에 추가해둠
        models.append(('ridge_' + str(weight), ridge))

        ridge.fit(x_train, y_train)
        pred = ridge.predict(x_valid)
        set_data('Ridge(alpha={}'.format(weight), pred, y_valid)

    print("\n", "=" * 3, "03. 엘라스틱넷 ( ElasticNt) - L1, L2 규제를 혼합하여 사용한 모델", "=" * 3)
    ratios = [0.2, 0.5, 0.8]
    for ratio in ratios:
        elasticnet = ElasticNet(alpha=0.5, l1_ratio=ratio , max_iter=1000)
        # 모델에 추가해둠
        models.append(('elasticnet_' + str(ratio), elasticnet))

        elasticnet.fit(x_train, y_train)
        pred = elasticnet.predict(x_valid)
        set_data('ElasticNet(alpha={}'.format(ratio), pred, y_valid)

    print("\n", "=" * 3, "04. 스케일러 적용된 엘라스틱넷 데이터 ", "=" * 3)
    elasticnet_pipline_std = make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=1000)
    )
    elasticnet_pipline_minmax = make_pipeline(
        MinMaxScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=1000)
    )
    elasticnet_pipline_rob = make_pipeline(
        RobustScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=1000)
    )

    models.append(('elasticnet_pipline_std', elasticnet_pipline_std))
    models.append(('elasticnet_pipline_minmax', elasticnet_pipline_minmax))
    models.append(('elasticnet_pipline_rob', elasticnet_pipline_rob))

    elasticnet_pred_std = elasticnet_pipline_std.fit(x_train, y_train).predict(x_valid)
    elasticnet_pred_minmax = elasticnet_pipline_minmax.fit(x_train, y_train).predict(x_valid)
    elasticnet_pred_rob = elasticnet_pipline_rob.fit(x_train, y_train).predict(x_valid)
    set_data('elasticnet_pred_std', elasticnet_pred_std, y_valid)
    set_data('elasticnet_pipline_minmax', elasticnet_pred_minmax, y_valid)
    set_data('elasticnet_pipline_rob', elasticnet_pred_rob, y_valid)

    print("\n", "=" * 3, "05. Polynomial 적용된 데이터 ", "=" * 3)

    poly_pipeline = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=1000)
    )

    models.append(('poly_pipeline', poly_pipeline))
    poly_pred = poly_pipeline.fit(x_train, y_train).predict(x_valid)
    set_data('poly_pipeline', poly_pred, y_valid)

    view_graph('poly_pipeline', poly_pred, y_valid)
    plt.show()

    return x_train, x_valid, y_train, y_valid, models


def ensemble_01():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            01. 보팅 (Voting) - 회귀 (Regression)
        Describe
            Voting은 단어 뜻 그대로 투표를 통해 결정하는 방식
            Bagging과 투표방식이라는 점에서 유사하지만, 다음과 같은 차이점이 있다.
                - Voting은 다른 알고리즘 model을 조합해서 사용한다.
                - Bagging은 같은 알고리즘 내에서 다른 sample 조합을 사용합니다.

            Voting은 반드시 Tuple 형태로 모델을 정의해야 합니다.

            https://scikit-learn.org/stable/modules/classes.html?highlight=ensemble#module-sklearn.ensemble
        sub Contents
            01.
    """
    print("\n", "=" * 5, "01. 보팅 (Voting) - 회귀 (Regression)", "=" * 5)

    # 데이터셋은 전에 활용했던 데이터 셋을 이용하여 사용
    from sklearn.ensemble import VotingRegressor, VotingClassifier
    x_train, x_valid, y_train, y_valid, models = compare_data_set()

    print("\n", "=" * 3, "01.", "=" * 3)
    print(models)

    voting_regression = VotingRegressor(models, n_jobs=-1)
    voting_regression.fit(x_train, y_train)
    voting_pred = voting_regression.predict(x_valid)
    set_data('voting ens', voting_pred, y_valid)
    view_graph('voting ens', voting_pred, y_valid)

    # voting_classifer = VotingClassifier(models, voting='hard', n_jobs=-1)
    # voting_classifer.fit(x_train, y_train)
    # voting_classifer_pred = voting_regression.predict(x_valid)
    # set_data('voting ens_soft', voting_classifer_pred, y_valid)
    # view_graph('voting ens', voting_classifer_pred, y_valid)

    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


ensemble_01()


def ensemble_02():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            02
        Describe

        sub Contents
            01.
    """
    print("\n", "=" * 5, "02", "=" * 5)
    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# ensemble_02()


def ensemble_temp():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
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

# ensemble_temp()


