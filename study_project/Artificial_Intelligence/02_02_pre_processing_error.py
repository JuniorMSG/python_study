"""
    subject
        Machine_Running
    topic
        preprocessing classification Error
    Describe
        데이터 오차 저리방법
    Contents
        01. 용어 정리, 정확도, 오차행렬
        02. 정밀도, 재현률, f1 score
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


def font_set():
    from matplotlib import font_manager, rc
    font_path = "C:\Windows\Fonts\HYGTRE.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font)
    # RuntimeWarning: Glyph 8722 missing from current font.
    plt.rc('axes', unicode_minus=False)


def preprocessing_error_01():
    """
        subject
            Machine_Running
        topic
            preprocessing classification Error

        content
            14. 용어 정리, 정확도, 오차행렬
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
            03. Confusion matrix (오차 행렬)

    """
    print("\n", "=" * 5, "01", "=" * 5)
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    import numpy as np
    cancer = load_breast_cancer()
    # print(cancer['DESCR'])
    df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df_cancer['target'] = cancer['target']

    pos = df_cancer.loc[df_cancer['target'] == 1]
    neg = df_cancer.loc[df_cancer['target'] == 0]

    # 357개
    print('pos :', pos.value_counts())

    # 212개
    print('neg :', neg.value_counts())

    print("\n", "=" * 3, "01. Accuracy (정확도)", "=" * 3)
    sample_general = pd.concat([pos, neg], sort=True)
    x_train, x_valid, y_train, y_valid = train_test_split(sample_general.drop('target', 1), sample_general['target'], random_state=42)
    model_general = LogisticRegression(max_iter=5000)
    model_general.fit(x_train, y_train)
    pred_general = model_general.predict(x_valid)

    print('일반적인 데이터', (pred_general == y_valid).mean())


    print("\n", "=" * 3, "02. 정확도의 역설 (Accuracy Paradox) 예시", "=" * 3)

    # Negative의 모수를 강제로 줄이면 긍정적인 데이터만 활용하게 된다.
    sample_divide = pd.concat([pos, neg[:5]], sort=True)
    x_train_divide, x_valid_divide, y_train_divide, y_valid_divide = train_test_split(sample_divide.drop('target', 1), sample_divide['target'], random_state=42)
    model_divide = LogisticRegression(max_iter=5000)
    model_divide.fit(x_train_divide, y_train_divide)
    pred_divide = model_divide.predict(x_valid_divide)

    print('일반적인 데이터', (pred_divide == y_valid_divide).mean())

    # 정확도를 높히기 위해서 테스트 케이스 데이터를 강제로 조정할 경우 더 높은 정확도를 얻게된다.
    pred_error = np.ones(shape=y_valid_divide.shape)
    print(pred_error)
    print('데이터 강제 조정시', (pred_error == y_valid_divide).mean())


    print("\n", "=" * 3, "03. Confusion matrix (오차 행렬)", "=" * 3)
    # 오차 행렬은 데이터를 확인하면서 정확도를 확인하는 용도로 사용하면 좋다.
    from sklearn.metrics import confusion_matrix
    font_set()
    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='data model')
    fig.suptitle('Confusion matrix (오차 행렬)', fontsize=15)


    axes[0, 0].set_title('정상데이터 모델')
    axes[1, 0].set_title('모수 줄인 데이터 모델')
    axes[1, 1].set_title('데이터 강제조정 모델')
    sns.heatmap(ax=axes[0, 0], data=confusion_matrix(y_valid, pred_general), annot=True, cmap='Reds')
    sns.heatmap(ax=axes[1, 0], data=confusion_matrix(y_valid_divide, pred_divide), annot=True, cmap='Reds')
    sns.heatmap(ax=axes[1, 1], data=confusion_matrix(y_valid_divide, pred_error), annot=True, cmap='Reds')
    plt.show()


# preprocessing_error_01()


def preprocessing_error_02():
    """
        subject
            Machine_Running
        topic
            preprocessing classification Error

        content
            02. 정밀도 (precision), 재현율 (recall), f1 score
        Describe
            정확도의 역설(함정)을 보완하기 위한 방법

        sub Contents
            01. 정밀도 (precision)
                TP / (TP + FP)
                무조건 양성으로 판단하면 좋은 정밀도를 얻기 때문에 유용하지 않다.

            02. 재현율 (recall)
                정확하게 감지한 양성 샘플의 비율
                민감도 (sensitivity) 혹은 True Positive Rate (TPR) 이라고도 불린다.
                TP / (TP + FN)

            03.  f1 score
                정밀도와 재현율의 조화 평균을 나타내는 지표
    """
    print("\n", "=" * 5, "02", "=" * 5)
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    cancer = load_breast_cancer()
    # print(cancer['DESCR'])
    df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df_cancer['target'] = cancer['target']

    pos = df_cancer.loc[df_cancer['target'] == 1]
    neg = df_cancer.loc[df_cancer['target'] == 0]

    sample_general = pd.concat([pos, neg], sort=True)
    x_train, x_valid, y_train, y_valid = train_test_split(sample_general.drop('target', 1), sample_general['target'], random_state=42)
    model_general = LogisticRegression(max_iter=5000)
    model_general.fit(x_train, y_train)
    pred_general = model_general.predict(x_valid)
    # import
    from sklearn.metrics import precision_score, recall_score, f1_score

    print("\n", "=" * 3, "01. 정밀도 (precision)", "=" * 3)
    print('정밀도 (precision)',  precision_score(y_valid, pred_general))


    print("\n", "=" * 3, "02. 재현율 (recall) ", "=" * 3)
    # 재현율 (recall)
    recall_score(y_valid, pred_general)
    print('재현율 (recall)',  precision_score(y_valid, pred_general))

    print("\n", "=" * 3, "03. f1 score", "=" * 3)
    print('f1_score', f1_score(y_valid, pred_general))


preprocessing_error_02()

def preprocessing_error_temp():
    """
        subject
            Machine_Running
        topic
            preprocessing classification Error
        content
            temp
        Describe

        sub Contents
            01.
    """
    print("\n", "=" * 5, "temp", "=" * 5)
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import numpy as np

    cancer = load_breast_cancer()
    # print(cancer['DESCR'])
    df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df_cancer['target'] = cancer['target']

    pos = df_cancer.loc[df_cancer['target'] == 1]
    neg = df_cancer.loc[df_cancer['target'] == 0]


    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# preprocessing_error_temp()


