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
            text_mining
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
    # 단어별 빈도 분석
    # 워드 클라우드 시각화
    # pip install pytagcloud pygame simplejson

    from collections import Counter
    import random
    import pytagcloud
    import webbrowser

    # dict 로 만든 워드 카운트를 넣는다.
    ranked_tags = Counter(word_count_dict).most_common(25)
    print(ranked_tags)

    taglist_01 = pytagcloud.make_tags(sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=True)[:40], maxsize=20)
    taglist_02 = pytagcloud.make_tags(Counter(word_count_dict).most_common(25), maxsize=60)
    pytagcloud.create_tag_image(taglist_01, 'wordcloud_example_01.jpg', rectangular=False)
    pytagcloud.create_tag_image(taglist_02, 'wordcloud_example_02.jpg', rectangular=False)

    # pip install iPython

    from PIL import Image
    # Image.open('wordcloud_example_01.jpg').show()
    # Image.open('wordcloud_example_02.jpg').show()
    print(ranked_tags)

    print("\n", "=" * 3, "03.", "=" * 3)

    # 장면별 중요 단어 시각화
    # TF-IDF 변환
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_vectorizer = TfidfTransformer()
    tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)
    print(tf_idf_vect.shape)
    print(tf_idf_vect)

    print(tf_idf_vect[0].toarray().shape)
    print(tf_idf_vect[0].toarray())

    # 백터 : 단어 맵핑
    invert_index_vectorizer = {v:k for k, v in vect.vocabulary_.items()}

    # 중요 단어 추출 Top 3 TF-IDF
    print(np.argsort(tf_idf_vect[0].toarray())[0][-3:])

    # 중요 단어 추출 모든단어 Top 3 TF-IDF
    print(np.argsort(tf_idf_vect.toarray())[:, -3:])

    top3_word = np.argsort(tf_idf_vect.toarray())[:, -3:]
    df['important_word_indexes'] = pd.Series(top3_word.tolist())
    print(df.head())

    def convert_to_word(x):
        word_list = []
        for word in x:
            word_list.append(invert_index_vectorizer[word])
        return word_list

    df['important_words'] = df['important_word_indexes'] .apply(lambda x : convert_to_word(x))

    print(df.head())


text_mining_01()


def text_mining_02():
    """
        subject
            Machine_Running
        topic
            text_mining
        content
            02. 감성 분류의 과정
        Describe
            1. 텍스트 데이터 전처리
            2. 이진 분류
            3. 긍/부정 키워드 분석

        sub Contents
            01.
    """
    print("\n", "=" * 5, "02", "=" * 5)

    df = pd.read_csv("./data_file/tripadviser_review.csv")
    print(df.shape)
    print(df.isnull().sum())
    print(df.info())
    print(len(df['text'].values.sum()))

    # 한국어 텍스트 데이터 전처리
    # pip install konlpy
    # SystemError: java.nio.file.InvalidPathException: Illegal char <*> at index 60: C:\Users\ngcc\.conda\envs\cub\lib\site-packages\konlpy\java\*
    # pip install jpype1 - error
    # https://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype에서 버전에 맞는 JPype 다운로드 ( JPype1-1.2.0-cp36-cp36m-win_amd64.whl)


    # konlpy pip install konlpy==0.5.1 jpype1 Jpype1-py3
    import re
    def apply_regular_expression(word):
        hangul = re.compile(('[^ ㄱ-ㅣ가-힣]'))
        result = hangul.sub('', word)
        return result
    print(df['text'][0])
    print(apply_regular_expression(df['text'][0]))

    # 한국어 형태소분석 - 명사 단위
    from konlpy.tag import Okt
    from collections import Counter

    nouns_tagger = Okt()
    nouns = nouns_tagger.nouns(apply_regular_expression(df['text'][0]))
    nouns = nouns_tagger.nouns(''.join(df['text'].tolist()))
    # print(nouns)
    counter = Counter(nouns)

    # 최대 10개 출력
    print(counter.most_common(10))

    # 한글자 명사 제거
    available_counter = Counter({x:counter[x] for x in counter if len(x) > 1})

    # 최대 10개 출력
    print(available_counter.most_common(10))

    # 불용어 사전
    stopwords = pd.read_csv('./data_file/korean_stopwords.txt').values.tolist()
    # print(stopwords)

    jeju_stopwords = ['제주', '제주도', '호텔', '리뷰', '숙소', '여행', '트립']
    for word in jeju_stopwords:
        stopwords.append(word)

    from sklearn.feature_extraction.text import CountVectorizer

    def text_preprocessing(word):
        # 3-1. 정규 표현식 처리
        hangul = re.compile(('[^ ㄱ-ㅣ가-힣]'))
        result = hangul.sub('', word)

        # 3-2. 형태소 분석
        tagger = Okt()
        nouns = nouns_tagger.nouns(result)

        # 3-3. 한글자 키워드 제거 , 불용어 처리
        nouns = [x for x in nouns if len(x) > 1]
        nouns = [x for x in nouns if x not in stopwords]

        return nouns

    vect = CountVectorizer(tokenizer=lambda x: text_preprocessing(x))
    bow_vect = vect.fit_transform(df['text'].tolist())
    word_list = vect.get_feature_names()
    count_list = bow_vect.toarray().sum(axis=0)

    print(bow_vect.shape)

    # word count dict 생성
    word_count_dict = dict(zip(word_list, count_list))
    print(str(word_count_dict))

    # 3-5 TF-IDF 변환
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_vectorizer = TfidfTransformer()
    tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)

    print(tf_idf_vect[0])

    # 백터 : 단어 맵핑
    invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}

    # 중요 단어 추출 Top 3 TF-IDF
    print(np.argsort(tf_idf_vect[0].toarray())[0][-3:])

    # 중요 단어 추출 모든단어 Top 3 TF-IDF
    print(np.argsort(tf_idf_vect.toarray())[:, -3:])

    top3_word = np.argsort(tf_idf_vect.toarray())[:, -3:]
    df['important_word_indexes'] = pd.Series(top3_word.tolist())
    print(df.head())

    def convert_to_word(x):
        word_list = []
        for word in x:
            word_list.append(invert_index_vectorizer[word])
        return word_list

    df['important_words'] = df['important_word_indexes'] .apply(lambda x : convert_to_word(x))

    print(df.head())

    # Logistic Regreesion 분류
    # 4-1 ) 데이터셋 생성
    print(df.sample(10))
    df.rating.hist()
    plt.show()

    def rating_to_label(rating):
        if rating > 3:
            return 1
        else:
            return 0
    df['y'] = df['rating'].apply(lambda x: rating_to_label(x))
    print(df.y.value_counts())


    # 데이터셋 분리
    from sklearn.model_selection import train_test_split
    y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(tf_idf_vect, y, test_size=0.3)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    lr = LogisticRegression(random_state=30)
    lr.fit(x_train, y_train)

    lr_pred = lr.predict(x_test)

    print(accuracy_score(y_test, lr_pred))
    print(precision_score(y_test, lr_pred))
    print(recall_score(y_test, lr_pred))
    print(f1_score(y_test, lr_pred))

    # recall score 조정을 위해서 샘플링 조절
    from sklearn.metrics import confusion_matrix
    confmat = confusion_matrix(y_test, lr_pred)
    print(confmat)

    positive_random_idx = df[df['y'] == 1].sample(275, random_state=30).index.tolist()
    negative_random_idx = df[df['y'] == 0].sample(275, random_state=30).index.tolist()

    random_idx = positive_random_idx + negative_random_idx
    x = tf_idf_vect[random_idx]
    y = df['y'][random_idx]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    lr = LogisticRegression(random_state=30)
    lr.fit(x_train, y_train)

    lr_pred = lr.predict(x_test)

    print(accuracy_score(y_test, lr_pred))
    print(precision_score(y_test, lr_pred))
    print(recall_score(y_test, lr_pred))
    print(f1_score(y_test, lr_pred))

    confmat = confusion_matrix(y_test, lr_pred)
    print(confmat)

    # 5. 긍정/부정 키워드 분석
    # Logistic Regression 모델으 coef 분석

    plt.rcParams['figure.figsize'] = [10, 8]
    plt.bar(range(len(lr.coef_[0])), lr.coef_[0])

    # 긍정 부정 인덱스 생성
    coef_pos_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=True)
    coef_neg_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse=False)

    # 백터 : 단어 맵핑
    invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}

    for coef in coef_pos_index[:15]:
        print(invert_index_vectorizer[coef[1]], coef[0])

    for coef in coef_neg_index[:15]:
        print(invert_index_vectorizer[coef[1]], coef[0])


    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# text_mining_02()


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


