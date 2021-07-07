"""
    subject
        Machine_Running
    topic
        회귀 (regression) 예측
        sklearn
        https://scikit-learn.org/stable/
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    Describe
        수치형 값을 예측한다.
        ex) 매출액, 주택 가격 예측, 주식 가격 예측등등

    Contents
        01. 용어 정리, 정확도, 오차행렬
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


def regression_01():
    """
        subject
            Machine_Running
        topic
            회귀 (regression) 예측
        content
            01. load_boston 데이터
        Describe
            Attribute Information (in order):
                - CRIM : 범죄율
                    per capita crime rate by town

                -  ZN : 25,000 평방 피트 당 주거용 토지의 비율
                    proportion of residential land zoned for lots over 25,000 sq.ft.
                - INDUS : 비소매 비즈니스 면적 비율
                    proportion of non-retail business acres per town
                - CHAS : 찰스 강 더미 변수 (하천을 향하면 1, 아니면 0)
                    Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                - NOX : 산화 질소 농도 (천만분의 1)
                    nitric oxides concentration (parts per 10 million)
                - RM : 주거 당 평균 객실 수
                    average number of rooms per dwelling
                - AGE : 1940년 이전에 건축된 자가 소유 점유 비율
                    proportion of owner-occupied units built prior to 1940
                - DIS : 5개의 보스턴 고용 센터까지의 가중 거리
                    weighted distances to five Boston employment centres
                - RAD : 고속도로 접근성 지수
                    index of accessibility to radial highways
                - TAX : 10,000달러 당 전체 가치 재산 세율
                    full-value property-tax rate per $10,000
                - PTRATIO : 도시 별 학생-교사 비율
                    pupil-teacher ratio by town
                - B : 1000(Bk - 0.63)^2 Bk는 도시 별 검정 비율
                1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
                - LSTAT : 인구의 낮은 지위
                    % lower status of the population
                - MEDV : 소유 주택의 중앙값 (1,000 달러 단위)
                    Median value of owner-occupied homes in $1000's
        sub Contents
            01. 데이터 만들기
            02. 평가지표 수식으로 직접 구해보기
            03. sklearn 평가지표 수식 사용해보기
    """

    print("\n", "=" * 5, "01", "=" * 5)

    print("\n", "=" * 3, "01. 데이터 만들기", "=" * 3)
    data = load_boston()
    df_boston = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_boston['MEDV'] = data['target']
    print(df_boston.head())
    x_train, x_valid, y_train, y_valid = train_test_split(df_boston.drop('MEDV', 1), df_boston['MEDV'])
    print(x_train.shape, x_valid.shape)

    print("\n", "=" * 3, "02. 평가지표 수식으로 직접 구해보기", "=" * 3)
    pred_value = np.array([3, 4, 5])
    actual_value = np.array([5, 10, 15])
    # MSE (Mean Squared Error)
    # 예측값과 실제값의 차이에 대한 제곱의 평균을 낸 값

    mse_value = ((pred_value - actual_value)**2).mean()
    print(mse_value)

    # MAE (Mean Abosolute ERror)
    # 예측값과 실제값의 차이에 대한 절대값에 평균을 낸 값
    mae_value = np.abs(pred_value - actual_value).mean()
    print(mae_value)

    # RMSE (Root Mean Squared Error)
    # 예측값과 실제값의 차이에 대한 제곱에 대하여 평균을 낸뒤 루트를 씌운값
    rmse_value = np.sqrt(mse_value)
    print(rmse_value)

    print("\n", "=" * 3, "03.", "=" * 3)
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1, normalize=False)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)

    print(mean_absolute_error(pred, y_valid))
    print(mean_squared_error(pred, y_valid))

    view_graph('Linear', pred, y_valid)


# regression_01()

def plot_coef(axes, name, columns, coef):
    coef_df = pd.DataFrame(list(zip(columns, coef)))
    coef_df.columns = ['feature', 'coef']
    coef_df = coef_df.sort_values('coef', ascending=False).reset_index(drop=True)

    axes.set_title(name)
    axes.barh(np.arange(len(coef_df)), coef_df['coef'])
    idx = np.arange(len(coef_df))
    axes.set_yticks(idx)
    axes.set_yticklabels(coef_df['feature'])


def regression_02():
    """
        subject
            Machine_Running
        topic
            회귀 (regression) 예측
        content
            02. 규제 (Regularization)
        Describe
            학습이 과대적합 되는 것을 방지하기 위해서 penalty를 부여하는 것
            Y = w * X + bias
            예측값 = 가중치 * 인풋 데이터 + bias

            L1 규제 (L1 Regularization)
                - 가중치의 제곱의 합이 아닌 가중치의 합을 더한 값에 규제 강도(Regularization Strength)를 곱하여 오차에 더한다.
                - 약점으로 어떤 가중치는 0이 되는등 모델에서 완전히 제외되는 특성이 생긴다.

            L2 규제 (L2 Regularization)
                - 각 가중치 제곱의 합에 규제 강도(Regularization Strength)를 곱한다.
                - 규제 강도를 크게 하면 가중치가 감소한다. (규제 중요도 Up)
                - 규제 강도를 작게 하면 가중치가 증가한다. (규재 중요도 Down)

            L2 규제가 L1 규제에 비해 더 안정적이라 일반적으로는 L2규제가 더 많이 사용된다.

            릿지 (Ridge) - L2 규제를 활용한 모델
                Error = MSE + a * w * w

            라쏘 (Lasso) - L1 규제를 활용한 모델
                Error = MSE + a * |w|

            엘라스틱넷 ( ElasticNt) - L1, L2 규제를 혼합하여 사용한 모델

        sub Contents
            01. 라쏘 (Lasso) - L1 규제를 활용한 모델
            02. 릿지 (Ridge) - L2 규제를 활용한 모델
            03. 엘라스틱넷 ( ElasticNt) - L1, L2 규제를 혼합하여 사용한 모델
    """
    print("\n", "=" * 5, "02. 규제 (Regularization)", "=" * 5)
    from sklearn.linear_model import Ridge, Lasso, ElasticNet

    data = load_boston()
    df_boston = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_boston['MEDV'] = data['target']
    print(df_boston.head())
    x_train, x_valid, y_train, y_valid = train_test_split(df_boston.drop('MEDV', 1), df_boston['MEDV'])
    print(x_train.shape, x_valid.shape)


    weights = [100, 10, 1, 0.1, 0.01, 0.001]
    print("\n", "=" * 3, "01. 라쏘 (Lasso) - L1 규제를 활용한 모델", "=" * 3)

    for weight in weights:
        lasso = Lasso(alpha=weight)
        lasso.fit(x_train, y_train)
        pred = lasso.predict(x_valid)
        set_data('Lasso(alpha={}'.format(weight), pred, y_valid)

    view_graph('Lasso', pred, y_valid)

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='view')
    # Regression coefficients 독립변수의 값에 영향을 미치는 회귀 계수

    lasso_100 = Lasso(alpha=100)
    lasso_100.fit(x_train, y_train)
    lasso_pred_100 = lasso_100.predict(x_valid)
    plot_coef(axes[0, 0], 'Lasso(alpha=100)',  x_train.columns, lasso_100.coef_)

    lasso_1 = Lasso(alpha=1)
    lasso_1.fit(x_train, y_train)
    lasso_pred_100 = lasso_1.predict(x_valid)
    plot_coef(axes[0, 1], 'Lasso(alpha=1)',  x_train.columns, lasso_1.coef_)

    lasso_01 = Lasso(alpha=0.1)
    lasso_01.fit(x_train, y_train)
    lasso_pred_01 = lasso_1.predict(x_valid)
    plot_coef(axes[1, 0], 'Lasso(alpha=0.1)',  x_train.columns, lasso_01.coef_)

    lasso_001 = Lasso(alpha=0.001)
    lasso_001.fit(x_train, y_train)
    lasso_pred_001 = lasso_001.predict(x_valid)
    plot_coef(axes[1, 1], 'Lasso(alpha=0.001)', x_train.columns, lasso_001.coef_)

    plt.show()


    print("\n", "=" * 3, "02. 릿지 (Ridge) - L2 규제를 활용한 모델", "=" * 3)

    for weight in weights:
        ridge = Ridge(alpha=weight)
        ridge.fit(x_train, y_train)
        pred = ridge.predict(x_valid)
        set_data('Ridge(alpha={}'.format(weight), pred, y_valid)

    view_graph('Ridge', pred, y_valid)

    print(x_train.columns)
    print(ridge.coef_)

    # Regression coefficients 독립변수의 값에 영향을 미치는 회귀 계수
    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='view')

    ridge_100 = Ridge(alpha=100)
    ridge_100.fit(x_train, y_train)
    ridge_pred_100 = ridge_100.predict(x_valid)
    plot_coef(axes[0, 0], 'Ridge(alpha=100)', x_train.columns, ridge_100.coef_)

    ridge_1 = Ridge(alpha=1)
    ridge_1.fit(x_train, y_train)
    ridge_pred_1 = ridge_1.predict(x_valid)
    plot_coef(axes[0, 1], 'Ridge(alpha=100)', x_train.columns, ridge_1.coef_)

    ridge_01 = Ridge(alpha=0.1)
    ridge_01.fit(x_train, y_train)
    ridge_pred_01 = ridge_01.predict(x_valid)
    plot_coef(axes[1, 0], 'Ridge(alpha=100)', x_train.columns, ridge_01.coef_)

    ridge_001 = Ridge(alpha=0.001)
    ridge_001.fit(x_train, y_train)
    ridge_pred_001 = ridge_001.predict(x_valid)
    plot_coef(axes[1, 1], 'Ridge(alpha=0.001)', x_train.columns, ridge_001.coef_)
    plt.show()

    print("\n", "=" * 3, "03. 엘라스틱넷 ( ElasticNt) - L1, L2 규제를 혼합하여 사용한 모델", "=" * 3)
    # l1_ratio (default = 0.5) 0에 가까울수록 L2 규제만 사용, 1에 가까울 수록 L1규제만 사용
    ratios = [0.2, 0.5, 0.8]

    for ratio in ratios:
        elasticnet = ElasticNet(alpha=0.5, l1_ratio=ratio)
        elasticnet.fit(x_train, y_train)
        pred = elasticnet.predict(x_valid)
        set_data('ElasticNet(alpha={}'.format(ratio), pred, y_valid)
    view_graph('ElasticNet', pred, y_valid)

    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='view')
    elasticnet_20 = ElasticNet(alpha=0.5, l1_ratio=0.2)
    elasticnet_20.fit(x_train, y_train)
    elasticnet_pred_20 = elasticnet_20.predict(x_valid)
    plot_coef(axes[0, 0], 'ElasticNet(alpha=0.5, l1_ratio=0.2)', x_train.columns, elasticnet_20.coef_)

    elasticnet_40 = ElasticNet(alpha=0.5, l1_ratio=0.4)
    elasticnet_40.fit(x_train, y_train)
    elasticnet_pred_001 = elasticnet_40.predict(x_valid)
    plot_coef(axes[0, 1], 'ElasticNet(alpha=0.5, l1_ratio=0.4)', x_train.columns, elasticnet_40.coef_)

    elasticnet_60 = ElasticNet(alpha=0.5, l1_ratio=0.6)
    elasticnet_60.fit(x_train, y_train)
    elasticnet_pred_001 = elasticnet_60.predict(x_valid)
    plot_coef(axes[1, 0], 'ElasticNet(alpha=0.5, l1_ratio=0.6)', x_train.columns, elasticnet_60.coef_)

    elasticnet_80 = ElasticNet(alpha=0.5, l1_ratio=0.8)
    elasticnet_80.fit(x_train, y_train)
    elasticnet_pred_001 = elasticnet_80.predict(x_valid)
    plot_coef(axes[1, 1], 'ElasticNet(alpha=0.5, l1_ratio=0.8)', x_train.columns, elasticnet_80.coef_)

    print(elasticnet_80.coef_)

    plt.show()


# regression_02()


def regression_03():
    """
        subject
            Machine_Running
        topic
            회귀 (regression) 예측
        content
            03. 파이프라인

            Scaler를 쉽게 적용하기 위해서 사용한다.
                StandardScaler  : 평균(mean)을 0으로, 표준편차(std)를 1로 만들어주는 스케일러
                MinMaxScaler    : min값과 max값을 0~1사이로 정규화화        Describe
                RobustScaler    : 중앙값(median)이 0, IQR(interquartile rane)이 1이 되도록 반환

        sub Contents
            01.
    """
    print("\n", "=" * 5, "03", "=" * 5)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.pipeline import make_pipeline

    data = load_boston()
    df_boston = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_boston['MEDV'] = data['target']
    x_train, x_valid, y_train, y_valid = train_test_split(df_boston.drop('MEDV', 1), df_boston['MEDV'])



    print("\n", "=" * 3, "01.", "=" * 3)
    std_scaler = StandardScaler()
    std_scaler = std_scaler.fit_transform(x_train)
    print(round(pd.DataFrame(std_scaler).describe(), 2))

    minmax_scaler = MinMaxScaler()
    minmax_scaler = minmax_scaler.fit_transform(x_train)
    print(round(pd.DataFrame(minmax_scaler).describe(), 2))

    robust_scaler = RobustScaler()
    robust_scaler = robust_scaler.fit_transform(x_train)
    print(round(pd.DataFrame(robust_scaler).describe(), 2))
    print(round(pd.DataFrame(robust_scaler).median(), 2))

    print("\n", "=" * 3, "02.", "=" * 3)
    elasticnet_pipline_std = make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.2)
    )

    elasticnet_pipline_minmax = make_pipeline(
        MinMaxScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.2)
    )
    elasticnet_pipline_rob = make_pipeline(
        RobustScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.2)
    )

    elasticnet_pred_std = elasticnet_pipline_std.fit(x_train, y_train).predict(x_valid)
    elasticnet_pred_minmax = elasticnet_pipline_minmax.fit(x_train, y_train).predict(x_valid)
    elasticnet_pred_rob = elasticnet_pipline_rob.fit(x_train, y_train).predict(x_valid)
    set_data('elasticnet_pred_std', elasticnet_pred_std, y_valid)
    set_data('elasticnet_pipline_minmax', elasticnet_pred_minmax, y_valid)
    set_data('elasticnet_pipline_rob', elasticnet_pred_rob, y_valid)
    view_graph('elasticnet_pipline_rob', elasticnet_pred_rob, y_valid)

    print(my_predictions_mse)
    print(my_predictions_mae)
    print("\n", "=" * 3, "03.", "=" * 3)


regression_03()

def regression_04():
    """
        subject
            Machine_Running
        topic
            회귀 (regression) 예측
        content
            04. Polynomial Features
        Describe
            다항식의 계수간 상호작용을 통해 새로운 feature를 생성합니다
            ex) [a, b] 2개의 feature가 존재한다고 가정하고,
            degree=2로 설정한다면, polynomial features 는 [1, a, b, a^2, ab, b^2]가 됩니다.

        sub Contents
            01.
    """
    print("\n", "=" * 5, "04. Polynomial Features", "=" * 5)
    data = load_boston()
    df_boston = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_boston['MEDV'] = data['target']
    x_train, x_valid, y_train, y_valid = train_test_split(df_boston.drop('MEDV', 1), df_boston['MEDV'])

    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    print("\n", "=" * 3, "01.", "=" * 3)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(x_train)[0]
    print(poly_features)

    print("\n", "=" * 3, "02.", "=" * 3)
    poly_pipeline = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        ElasticNet(alpha=0.1, l1_ratio=0.2)
    )
    poly_pred = poly_pipeline.fit(x_train, y_train).predict(x_valid)
    set_data('poly_pipeline', poly_pred, y_valid)
    view_graph('poly_pipeline', poly_pred, y_valid)

    print("\n", "=" * 3, "03.", "=" * 3)


regression_04()


def regression_temp():
    """
        subject
            Machine_Running
        topic
            회귀 (regression) 예측
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

# regression_temp()


