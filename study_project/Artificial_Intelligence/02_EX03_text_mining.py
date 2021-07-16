"""
    subject
        Machine_Running
    topic
        텍스트 마이닝

    Describe

    Contens
        01.
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

def text_mining_01():
    """
        subject
            Machine_Running
        topic
            text_mining_01
        content
            01
        Describe
            텍스트 마이닝
                비정형 데이터임.
                데이터 분석에 활용 가능하도록 정형 데이터로 변경해줘야함
                - 원 핫 인코딩, 멀티 바이너라이저 등등..

                EX) 워드 클라우드 시각화
                EX) 감성 분류
                토픽 모델링 LDA
                Semantic Analysis - 기계가 백터단위로

            텍스트 데이터의 처리 방법 - 계산 가능한 데이터로 처리하는 방법

            BoW (Bag of Words)
                불용어가 아닌 형태소 추출
                각 형태소에 등장 횟수를 센다.
                이것을 백터화 한다.

            문서 단어 행렬 (Document-Term Matrix, DTM)
                - 어떤 것이 중요한 단어인지 알 수 없음.
            단어의 중요도를 계산하는 방법 TF-IDF (Term Frequency-Omverse Document Frequency)
            지금 문장에서 등장한 단어가 다른 문서에서 등장하지 않으면 더 중요한다는 가설로 계산한 수식임.
                TF : 특정 문서에서 특정 단어의 등장 횟수
                DF : 특정 단어가 등장한 문서의 수
                IDF : DF와 반비례 값을 가지는 수식
                TF-IDF : TF와 IDF를 곱한 값


        sub Contents
            01.
    """
    print("\n", "=" * 5, "01. 텍스트 데이터 전처리 실습", "=" * 5)
    df = pd.read_csv("./data_file/bourne_scenario.csv")
    print(df.head())
    print('shape', df.shape)
    print('결측치', df.isnull().sum())
    print('정보', df.info())

    # 정규 표현식 적용
    import re

    def apply_regular_expression(text):
        text = text.lower()
        english = re.compile('[^ a-z]')
        result = english.sub('', text)

        # 2개 이상의 공백 제거.
        result = re.sub(' +', ' ', result)
        return result

    df['prep_text'] = df['text'].apply(lambda x: apply_regular_expression(x))

    print(df.head())

    print("\n", "=" * 3, "01. BoW (Bag of Words)", "=" * 3)

    # 말뭉치(코퍼스) 생성 - 텍스트 데이터를 통째로 가져온것.
    corpus = df['prep_text'].tolist()
    print(corpus)

    # BoW 백터 생성
    from sklearn.feature_extraction.text import CountVectorizer

    vect = CountVectorizer(tokenizer=None, stop_words='english', analyzer='word').fit(corpus)
    bow_vect = vect.fit_transform(corpus)

    word_list = vect.get_feature_names()
    count_list = bow_vect.toarray().sum(axis=0)

    print(word_list[:5])
    print(count_list[:5])

    print(bow_vect.shape)

    print(bow_vect.toarray().sum(axis=0))

    word_count_dict = dict(zip(word_list, count_list))
    print(str(word_count_dict)[:100])

    import operator
    print(sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=True)[:5])

    # 단어 분포 탐색
    plt.hist(list(word_count_dict.values()))
    plt.show()


    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

text_mining_01()

def text_mining_temp():
    """
        subject
            Machine_Running
        topic
            text_mining_temp
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

# text_mining_temp()


