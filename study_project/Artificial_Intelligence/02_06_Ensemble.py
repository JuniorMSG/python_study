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

        1. 앙상블은 대체적으로 단일 모델 대비 성능이 좋다.
        2. 앙상블은 앙상블하는 기업인 Stacking, Weighted Blending도 참고해볼만함.
        3. 앙상블 모델은 적절한 Hyperparameter 튜닝이 중요하다.
        4. 앙상블 모델은 대체적으로 학습시간이 더 오래 걸린다.
        5. 모델 튜닝을 하는 데에 시간이 오래걸린다.

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

data = load_boston()
df_boston = pd.DataFrame(data['data'], columns=data['feature_names'])
df_boston['MEDV'] = data['target']
x_train, x_valid, y_train, y_valid = train_test_split(df_boston.drop('MEDV', 1), df_boston['MEDV'])

my_predictions_mse = {}
my_predictions_mae = {}
models = {}
models_pred = {}
colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
          ]

def last_graph(name, pred, y_valid):
    view_graph(name, pred, y_valid)

def plot_predictions(name_, pred, actual, axes):
    df = pd.DataFrame({'prediction': pred, 'actual': actual})
    df = df.sort_values(by='actual').reset_index(drop=True)

    sns.scatterplot(ax=axes, data=df['prediction'], marker='x', color='r')
    sns.scatterplot(ax=axes, data=df['actual'], alpha=0.7, marker='o', color='black')

    axes.set_title(name_, fontsize=15)
    axes.legend(['prediction', 'actual'], fontsize=12)


def set_data(name_, pred, actual, model):
    global my_predictions_mse
    global my_predictions_mae
    global models

    # MSE 값 측정
    mse = mean_squared_error(pred, actual)
    # MAE 값 측정
    mae = mean_absolute_error(pred, actual)

    my_predictions_mse[name_] = mse
    my_predictions_mae[name_] = mae
    models[name_] = model
    models_pred[name_] = pred

    y_value = sorted(my_predictions_mse.items(), key=lambda x: x[1], reverse=True)
    df_mse = pd.DataFrame(y_value, columns=['model', 'mse'])
    y_value = sorted(my_predictions_mae.items(), key=lambda x: x[1], reverse=True)
    df_mae = pd.DataFrame(y_value, columns=['model', 'mae'])

    return df_mse, df_mae


def get_models(model_count):
    models_grade = sorted(my_predictions_mse.items(), key=lambda x:x[1])
    model_list = np.array(models_grade[:model_count])[:, 0]

    get_model_list = []
    for model in model_list:
        get_model_list.append((model, models[model]))
    return get_model_list

def get_models_pred(model_count):
    models_grade = sorted(my_predictions_mse.items(), key=lambda x:x[1])
    model_list = np.array(models_grade[:model_count])[:, 0]

    get_model_pred_list = []
    for model in model_list:
        get_model_pred_list.append((model, models_pred[model]))
    return get_model_pred_list

def get_data(name_, pred, actual):
    global my_predictions_mse
    global my_predictions_mae
    global models

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
    axes[0, 1] = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]

    plot_predictions(name_, pred, actual, ax1)

    df_mse, df_mae = get_data(name_, pred, actual)
    min_ = df_mse['mse'].min() - 10
    max_ = df_mse['mse'].max() + 10

    ax2.set_yticks(np.arange(len(df_mse)))
    ax2.set_yticklabels(df_mse['model'], fontsize=15)
    ax2.set_title('MSE Error', fontsize=18)

    bars = ax2.barh(np.arange(len(df_mse)), df_mse['mse'])

    for i, v in enumerate(df_mse['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax2.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    ax2.set_xlim(min_, max_)

    min_ = df_mae['mae'].min() - 10
    max_ = df_mae['mae'].max() + 10

    ax3.set_yticks(np.arange(len(df_mae)))
    ax3.set_yticklabels(df_mae['model'], fontsize=15)
    ax3.set_title('MAE ERROR', fontsize=18)

    bars = ax3.barh(np.arange(len(df_mae)), df_mae['mae'])

    for i, v in enumerate(df_mae['mae']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax3.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    ax3.set_xlim(min_, max_)

    plt.tight_layout(True)
    plt.show()

def compare_data_set():
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.pipeline import make_pipeline

    global x_train, x_valid, y_train, y_valid, models


    # 비교군이 너무 많아서 가중치는 4개로 줄임.
    weights = [100, 1, 0.1, 0.001]

    print("\n", "=" * 3, "01. 라쏘 (Lasso) - L1 규제를 활용한 모델", "=" * 3)
    for weight in weights:
        lasso = Lasso(random_state=42, alpha=weight, max_iter=1000)
        lasso.fit(x_train, y_train)
        pred = lasso.predict(x_valid)
        set_data('Lasso(alpha={}'.format(weight), pred, y_valid, lasso)

    print("\n", "=" * 3, "02. 릿지 (Ridge) - L2 규제를 활용한 모델", "=" * 3)
    for weight in weights:
        ridge = Ridge(random_state=42, alpha=weight , max_iter=1000)
        ridge.fit(x_train, y_train)
        pred = ridge.predict(x_valid)
        set_data('Ridge(alpha={}'.format(weight), pred, y_valid, ridge)

    print("\n", "=" * 3, "03. 엘라스틱넷 ( ElasticNt) - L1, L2 규제를 혼합하여 사용한 모델", "=" * 3)
    ratios = [0.2, 0.5, 0.8]
    for ratio in ratios:
        elasticnet = ElasticNet(random_state=42, alpha=0.5, l1_ratio=ratio , max_iter=1000)
        # 모델에 추가해둠
        elasticnet.fit(x_train, y_train)
        pred = elasticnet.predict(x_valid)
        set_data('ElasticNet(alpha={}'.format(ratio), pred, y_valid, elasticnet)

    print("\n", "=" * 3, "04. 스케일러 적용된 엘라스틱넷 데이터 ", "=" * 3)
    elasticnet_pipline_std = make_pipeline(
        StandardScaler(),
        ElasticNet(random_state=42, alpha=0.1, l1_ratio=0.2, max_iter=1000)
    )
    elasticnet_pipline_minmax = make_pipeline(
        MinMaxScaler(),
        ElasticNet(random_state=42, alpha=0.1, l1_ratio=0.2, max_iter=1000)
    )
    elasticnet_pipline_rob = make_pipeline(
        RobustScaler(),
        ElasticNet(random_state=42, alpha=0.1, l1_ratio=0.2, max_iter=1000)
    )

    elasticnet_pred_std = elasticnet_pipline_std.fit(x_train, y_train).predict(x_valid)
    elasticnet_pred_minmax = elasticnet_pipline_minmax.fit(x_train, y_train).predict(x_valid)
    elasticnet_pred_rob = elasticnet_pipline_rob.fit(x_train, y_train).predict(x_valid)
    set_data('elasticnet_pred_std', elasticnet_pred_std, y_valid, elasticnet_pipline_std)
    set_data('elasticnet_pipline_minmax', elasticnet_pred_minmax, y_valid, elasticnet_pipline_minmax)
    set_data('elasticnet_pipline_rob', elasticnet_pred_rob, y_valid , elasticnet_pipline_rob)

    print("\n", "=" * 3, "05. Polynomial 적용된 데이터 ", "=" * 3)

    poly_pipeline = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=1000)
    )

    poly_pred = poly_pipeline.fit(x_train, y_train).predict(x_valid)
    set_data('poly_pipeline', poly_pred, y_valid, poly_pipeline)

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
    global x_train, x_valid, y_train, y_valid, models

    print("\n", "=" * 3, "01.", "=" * 3)
    print(models)

    voting_regression = VotingRegressor(models, n_jobs=-1)
    voting_regression.fit(x_train, y_train)
    voting_pred = voting_regression.predict(x_valid)
    set_data('voting ens', voting_pred, y_valid, voting_regression)
    # view_graph('voting ens', voting_pred, y_valid)

    # voting_classifer = VotingClassifier(models, voting='hard', n_jobs=-1)
    # voting_classifer.fit(x_train, y_train)
    # voting_classifer_pred = voting_regression.predict(x_valid)
    # set_data('voting ens_soft', voting_classifer_pred, y_valid)
    # view_graph('voting ens', voting_classifer_pred, y_valid)

    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


# ensemble_01()


def ensemble_02():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            02. Bagging
        Describe
            Bagging = Bootstrap Aggregating의 줄임말.
            다양한 샘플링 조합으로 단일 알고리즘 예측을 한 다음에 그거에 대해서 Ensemble 한다.

            Bootstrap = Sample(샘플) + Aggregating = 합산
            Bootstrap은 여러 개의 dataset을 중첩을 허용하게 하여 샘플링하여 분할하는 방식.
            EX) 데이터 셋의 구성이 1,2,3,4,5 이면
            [1, 2, 3] , [1, 3, 4], [2, 3, 5]로 ensemble 함.

            Voting은 여러 알고리즘의 조합에 대한 앙상블
            Bagging은 하나의 단일 알고리즘에 대하여 여러 개의 샘플 조합으로 앙상블

        sub Contents
            01. RandomForest
                DecisionTree(트리)기반 Bagging 앙상블
                굉장히 인기있는 앙상블 모델
                사용성이 쉽고 성능도 우수함
    """
    print("\n", "=" * 5, "02", "=" * 5)
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # 데이터셋은 전에 활용했던 데이터 셋을 이용하여 사용
    global x_train, x_valid, y_train, y_valid, models

    print("\n", "=" * 3, "01.", "=" * 3)
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(x_train,y_train)
    rfr_pred = rfr.predict(x_valid)
    set_data('RF Ensemble', rfr_pred, y_valid, rfr)
    # view_graph('RF Ensemble', rfr_pred, y_valid)

    # random_state          : 랜덤 시드 고정 값, 고정해두고 튜닝할 것
    # n_jobs                : CPU 사용 갯수
    # max_depth             : 깊어질 수 있는 최대 깊이, 과대적합 방지용
    # n_estimators          : 앙상블하는 트리의 개수
    # max_features          : 최대로 사용할 feature의 개수, 과대적합 방지용
    # min_samples_splits    : 트리가 분할할 때 최소 샘플의 갯수 default=2, 과대적합 방지용
    print("\n", "=" * 3, "02.", "=" * 3)


    rfr = RandomForestRegressor(random_state=42, n_estimators=1000, max_depth=7, max_features=0.8)
    rfr.fit(x_train,y_train)
    rfr_pred = rfr.predict(x_valid)
    set_data('RF n_estimators=1000 Ensemble', rfr_pred, y_valid, rfr)
    print("\n", "=" * 3, "03.", "=" * 3)
    rfr = RandomForestRegressor(random_state=42, n_estimators=500, max_depth=7, max_features=0.8)
    rfr.fit(x_train,y_train)
    rfr_pred = rfr.predict(x_valid)
    set_data('RF param Ensemble', rfr_pred, y_valid, rfr)
    # view_graph('RF n_estimators=500, Ensemble', rfr_pred, y_valid)


ensemble_02()


def ensemble_03():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            03. 부스팅 (Boosting)
        Describe
            약한 학습기를 순차적으로 학습을 하되, 이전 학습에 대하여 잘못 예측된 데이터에
            가중치를 부여해 오차를 보완해 나가는 방식입니다.
            대표적인 Boosting Ensemble
                1. AdaBoost
                2. GradientBoost
                    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
                3. LightGBM(LGBM)
                4. XGBoost
            장점
                성능이 매우 우수하다 (Lgbm, XGBoost)
        sub Contents
            단점
                부스팅 알고리즘의 특성상 계속 약점(오분류/잔차)을 보완하려고 하기 때문에 잘못된 레이블링이나 아웃라이어에
                필요 이상으로 민감할 수 있따.
                다른 앙상블 대비 학습 시간이 오래걸린다는 단점이존재한다.

            01. GradientBoost
    """
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    print("\n", "=" * 5, "03", "=" * 5)

    print("\n", "=" * 3, "01.", "=" * 3)

    # 데이터셋은 전에 활용했던 데이터 셋을 이용하여 사용
    global x_train, x_valid, y_train, y_valid, models

    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(x_train, y_train)
    gbr_pred = gbr.predict(x_valid)
    set_data('GradientBoost Ensemble', gbr_pred, y_valid, gbr)

    # Hyperparameter
    # random_state 랜덤 시드 고정 값, 고정해두고 튜닝
    # n_jobs : CPU 사용 개수
    # learning_rate 학습율, 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다. 적절한 값을 찾아야함. n_estimators와 같이 튜닝
    # default = 0.1
    # n_estimators : 부스팅 스테이지 수 (랜덤 포레스트 트리의 갯수 설정과 비슷한 개념 default = 100)
    # learning_rate * n_estimators를 같은 값으로 유지하는게 좋음
    # subsample : 샘플 사용 비율 (max_features와 비슷한 개념) 과대적합 방지용
    # min_samples_split : 노드 분할시 최소 샘플의 갯수 default =2 . 과대적합 방지용
    gbr = GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=200)
    gbr.fit(x_train, y_train)
    gbr_pred = gbr.predict(x_valid)
    set_data('Gradient 0.05 / 200', gbr_pred, y_valid, gbr)

    gbr = GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=200, subsample=0.8)
    gbr.fit(x_train, y_train)
    gbr_pred = gbr.predict(x_valid)
    set_data('Gradient 0.05 / 200 subsample', gbr_pred, y_valid, gbr)

    gbr = GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=200, subsample=0.8, min_samples_split=3)
    gbr.fit(x_train, y_train)
    gbr_pred = gbr.predict(x_valid)
    set_data('Gradient 0.05 / 200 min3', gbr_pred, y_valid, gbr)

    # last_graph('GradientBoost Ensemble', gbr_pred, y_valid)


ensemble_03()

def ensemble_04():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            04. XGBoost
        Describe
            eXtreme Gradient Boosting
            - scikit-learn 패키지 아님
            - 성능이 우수함
            - GBM 보다 빠르고 성능 향상됨.
            - 학습시간이 매우 느림
            # pip install xgboost 100MB가량됨.
            # GPU버전으로 사용시 GPU도 사용 가능함

        sub Contents
            01.
    """
    print("\n", "=" * 5, "04", "=" * 5)

    # 데이터셋은 전에 활용했던 데이터 셋을 이용하여 사용
    global x_train, x_valid, y_train, y_valid, models
    print("\n", "=" * 3, "01.", "=" * 3)
    from xgboost import XGBRegressor, XGBClassifier
    xgb = XGBRegressor(random_state=42)
    xgb.fit(x_train, y_train)
    xgb_pred = xgb.predict(x_valid)
    set_data('XGBoost Def', xgb_pred, y_valid, xgb)


    print("\n", "=" * 3, "02.", "=" * 3)
    # Hyperparameter
    # random_state 랜덤 시드 고정 값, 고정해두고 튜닝
    # n_jobs : CPU 사용 개수
    # learning_rate 학습율, 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다. 적절한 값을 찾아야함. n_estimators와 같이 튜닝
    # default = 0.1
    # n_estimators : 부스팅 스테이지 수 (랜덤 포레스트 트리의 갯수 설정과 비슷한 개념 default = 100)
    # learning_rate * n_estimators를 같은 값으로 유지하는게 좋음
    # subsample : 샘플 사용 비율 (max_features와 비슷한 개념) 과대적합 방지용
    # min_samples_split : 노드 분할시 최소 샘플의 갯수 default =2 . 과대적합 방지용

    xgb = XGBRegressor(random_state=42, learning_rate=0.025, n_estimators=400, subsample=0.8, max_features=0.1, max_depth=3)
    xgb.fit(x_train, y_train)
    xgb_pred = xgb.predict(x_valid)
    set_data('XGBoost max_depth 3', xgb_pred, y_valid, xgb)

    # last_graph('XGBoost Def', xgb_pred, y_valid)

    print("\n", "=" * 3, "03.", "=" * 3)

ensemble_04()


def ensemble_05():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            05. LightGBM
        Describe
            - scikit-learn 패키지 아님
            - 성능이 우수함
            - XGBoost 보다 속도가 빠름.
            pip install lightgbm
        sub Contents
            01.
    """
    print("\n", "=" * 5, "05", "=" * 5)
    from lightgbm import LGBMRegressor, LGBMClassifier
    global x_train, x_valid, y_train, y_valid, models
    print("\n", "=" * 3, "01.", "=" * 3)
    lgbm = LGBMRegressor(random_state=42)
    lgbm.fit(x_train, y_train)
    lgbm_pred = lgbm.predict(x_valid)
    set_data('LightGBM Def', lgbm_pred, y_valid, lgbm)

    print("\n", "=" * 3, "02.", "=" * 3)
    # Hyperparameter
    # random_state 랜덤 시드 고정 값, 고정해두고 튜닝
    # n_jobs : CPU 사용 개수
    # learning_rate 학습율, 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다. 적절한 값을 찾아야함. n_estimators와 같이 튜닝
    # default = 0.1
    # n_estimators : 부스팅 스테이지 수 (랜덤 포레스트 트리의 갯수 설정과 비슷한 개념 default = 100)
    # learning_rate * n_estimators를 같은 값으로 유지하는게 좋음
    # colsample_bytree : 샘플 사용 비율 (max_features, subSample와 비슷한 개념) 과대적합 방지용

    # lgbm = LGBMRegressor(random_state=42, learning_rate=0.05, n_estimators=200, colsample_bytree=0.8, subsample=0.8, max_depth=3)
    # lgbm.fit(x_train, y_train)
    # lgbm_pred = lgbm.predict(x_valid)
    # set_data('LightGBM 1', lgbm_pred, y_valid)
    #
    # lgbm = LGBMRegressor(random_state=42, learning_rate=0.05, n_estimators=200, colsample_bytree=0.8, subsample=0.8, max_depth=5)
    # lgbm.fit(x_train, y_train)
    # lgbm_pred = lgbm.predict(x_valid)
    # set_data('LightGBM 2', lgbm_pred, y_valid)

    lgbm = LGBMRegressor(random_state=42, learning_rate=0.05, n_estimators=200, colsample_bytree=0.8, subsample=0.8, max_depth=7)
    lgbm.fit(x_train, y_train)
    lgbm_pred = lgbm.predict(x_valid)
    set_data('LightGBM 3', lgbm_pred, y_valid, lgbm)

    # last_graph('LightGBM Def', lgbm_pred, y_valid)


    print("\n", "=" * 3, "03.", "=" * 3)


ensemble_05()


def ensemble_06():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            06. Stacking
        Describe
            개별 모델이 예측한 데이터를 기반으로 final_estimator 종합하여 예측을 수행

            - 성능을 극으로 끌어올릴 때 활용하기도 한다.
            - 과대적합을 유발할 수 있다. (데이터셋이 적은경우)
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
        sub Contents
            01.
    """
    print("\n", "=" * 5, "06", "=" * 5)
    from sklearn.ensemble import StackingRegressor
    global x_train, x_valid, y_train, y_valid, models
    print("\n", "=" * 3, "01.", "=" * 3)

    # 모델에서 MSE 값 낮은 순서대로 5개 추출
    models_grade = sorted(my_predictions_mse.items(), key=lambda x:x[1])
    model_list = np.array(models_grade[:5])[:, 0]

    get_model_list = []
    for model in model_list:
        get_model_list.append((model, models[model]))

    stack_reg = StackingRegressor(get_model_list, final_estimator=get_model_list[0][1], n_jobs=-1)
    stack_reg.fit(x_train, y_train)
    stack_pred = stack_reg.predict(x_valid)
    set_data('StackingRegressor', stack_pred, y_valid, stack_reg)
    # last_graph('StackingRegressor ', stack_pred, y_valid)


    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

ensemble_06()


def ensemble_07():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            07. Weighted Blending
        Describe
            각 모델의 예측값에 대하여 weight를 곱하여 최종 output 계산
            모델에 대한 가중치를 조절하여, 최종 output을 산출합니다.
            가중치의 합은 1.0이 되도록 합니다.
        sub Contents
            01.
    """
    print("\n", "=" * 5, "07", "=" * 5)
    from sklearn.ensemble import StackingRegressor
    global x_train, x_valid, y_train, y_valid, models
    print("\n", "=" * 3, "01.", "=" * 3)

    models_pred_data = get_models_pred(5)

    final_prediction = models_pred_data[0][1] * 0.3
    final_prediction += models_pred_data[1][1] * 0.25
    final_prediction += models_pred_data[2][1] * 0.2
    final_prediction += models_pred_data[3][1] * 0.15
    final_prediction += models_pred_data[4][1] * 0.1


    set_data('fianl_prediction 1', final_prediction, y_valid, 'final models')


    final_prediction = models_pred_data[0][1] * 0.2
    final_prediction += models_pred_data[1][1] * 0.2
    final_prediction += models_pred_data[2][1] * 0.2
    final_prediction += models_pred_data[3][1] * 0.2
    final_prediction += models_pred_data[4][1] * 0.2
    set_data('final_prediction 2', final_prediction, y_valid, 'final models2')
    # last_graph('final_prediction 2', final_prediction, y_valid)

    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


# compare_data_set()
ensemble_07()


def ensemble_08():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            08. Cross Validation
        Describe
            Cross Validation이란 모델을 평가하는 하나의 방법
            K-fold Cross Validation을 많이 활용한다.

            K-fold Cross Validation
                - K-겹 교차 검증은 모든 데이터가 최소 한 번은 데스트셋으로 쓰이도록 합니다.
        sub Contents
            01.
    """
    print("\n", "=" * 5, "08", "=" * 5)
    from sklearn.model_selection import KFold
    from lightgbm import LGBMRegressor, LGBMClassifier

    global x_train, x_valid, y_train, y_valid, models

    n_splits = 5
    KFold = KFold(n_splits=n_splits)

    X = np.array(df_boston.drop('MEDV', 1))
    Y = np.array(df_boston['MEDV'])

    lgbm_fold = LGBMRegressor(random_state=42)

    i = 1
    total_error = 0
    for train_index, test_index in KFold.split(X):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = Y[train_index], Y[test_index]
        lgbm_pred_fold = lgbm_fold.fit(x_train_fold, y_train_fold).predict(x_test_fold)
        error = mean_squared_error(lgbm_pred_fold, y_test_fold)
        print('Fold ={} score={:.2f}'.format(i, error))
        total_error += error
        i += 1
    print('Average ERror %s' % (total_error / n_splits))
    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)


ensemble_08()


def ensemble_09():
    """
        subject
            Machine_Running
        topic
            앙상블 (Ensemble) 예측
        content
            09. Hyperparameter 튜닝을 돕는 클래스 2가지
                RandomizedSearchCV, GridSearchCV

        Describe
            Cross Validation이란 모델을 평가하는 하나의 방법
            RandomizedSearchCV
                - 모든 매개 변수값이 시도되는 것이 아니라 지정된 분포에서 고정 된 수의 매개 변수 설정이 샘플링 된다.
                - 시도 된 매개 변수 설정의 수는 n_iter에 의해 제공된다.
            GridSearchCV
                - 모든 매개 변수 값에 대하여 완전 탐색을 시도합니다.
                - 최적화할 parameter가 많다면 시간이 매우 오래 걸립니다.

            주요 Hyperparameter
            random_state: 랜덤 시드 고정 값. 고정해두고 튜닝할 것!
            n_jobs: CPU 사용 갯수
            learning_rate:  학습율. 너무 큰 학습율은 성능을 떨어뜨리고, 너무 작은 학습율은 학습이 느리다.
                            적절한 값을 찾아야함. n_estimators와 같이 튜닝. default=0.1
            n_estimators: 부스팅 스테이지 수. (랜덤포레스트 트리의 갯수 설정과 비슷한 개념). default=100
            max_depth: 트리의 깊이. 과대적합 방지용. default=3.
            colsample_bytree: 샘플 사용 비율 (max_features와 비슷한 개념). 과대적합 방지용. default=1.0

        sub Contents
            01.
    """
    print("\n", "=" * 5, "09", "=" * 5)
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import GridSearchCV
    from lightgbm import LGBMRegressor, LGBMClassifier

    params = {
        'n_estimators': [200, 500, 1000, 2000],
        'learning_rate': [0.1, 0.05, 0.01, 0.005],
        'max_depth': [3, 4, 5, 6, 7],
        'colsample_bytree': [0.8, 0.9, 1],
        'subsample': [0.8, 0.9, 1]
    }

    global x_train, x_valid, y_train, y_valid, models

    # n_iter 총 몇번의 랜덤한 조합값을 만들어내라
    # cv=3 3개의 cv로 구성한다
    clf = RandomizedSearchCV(LGBMRegressor(), params, random_state=42, cv=3, n_iter=25, scoring='neg_mean_squared_error')
    clf.fit(x_train, y_train)

    lgbm_best = LGBMRegressor(random_state=42, n_estimators=2000, max_depth=3, learning_rate=0.005, colsample_bytree=0.9)
    lgbm_best_pred = lgbm_best.fit(x_train, y_train).predict(x_valid)
    set_data('RandomizedSearchCV lgbm_best', lgbm_best_pred, y_valid, 'RandomizedSearchCV lgbm_best')
    # last_graph('lgbm_best_pred', lgbm_best_pred, y_valid)

    print(clf.best_score_)
    print(clf.best_params_)

    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)

    # grid_search = GridSearchCV(LGBMRegressor(), params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    # grid_search_pred = grid_search.fit(x_train, y_train).predict(x_valid)
    # {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}

    # print(grid_search.best_score_)
    # print(grid_search.best_params_)

    lgbm_best = LGBMRegressor(random_state=42, subsample=0.8, n_estimators=200, max_depth=5, learning_rate=0.1, colsample_bytree=1)
    lgbm_best_pred = lgbm_best.fit(x_train, y_train).predict(x_valid)
    set_data('GridSearchCV lgbm_best', lgbm_best_pred, y_valid, 'GridSearchCV lgbm_best')
    last_graph('GridSearchCV lgbm_best', lgbm_best_pred, y_valid)

    print("\n", "=" * 3, "03.", "=" * 3)

ensemble_09()



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


