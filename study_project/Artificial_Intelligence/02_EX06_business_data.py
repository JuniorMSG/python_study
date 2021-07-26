"""
    subject
        Machine_Running
    topic
        비즈니스 데이터 실습

    Describe

        axes : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
        x = np.linspace(0, 2 * np.pi, 400)
        y = np.sin(x ** 2)
        ax_temp.scatter(x, y)

    Contens
        01.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager, rc

# pip install missingno==0.4.2
# pip install squarify==0.4.3

def font_set():
    font_path = "C:\Windows\Fonts\HYGTRE.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font)
    plt.rc('axes', unicode_minus=False)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], label='한글테스트용')
    plt.legend()
    plt.show()


def business_01():
    """
        subject
            Machine_Running
        topic
            EX6. 비즈니스 데이터 실습
        content
            01
        Describe

        sub Contents
            01.
    """

    print("\n", "=" * 5, "01", "=" * 5)
    font_set()
    df_customers                = pd.read_csv("data_file/business/olist_customers_dataset.csv")
    df_geolocation              = pd.read_csv("data_file/business/olist_geolocation_dataset.csv")
    df_order_items              = pd.read_csv("data_file/business/olist_order_items_dataset.csv")
    df_order_payments           = pd.read_csv("data_file/business/olist_order_payments_dataset.csv")
    df_order_reviews            = pd.read_csv("data_file/business/olist_order_reviews_dataset.csv")
    df_order                    = pd.read_csv("data_file/business/olist_orders_dataset.csv")
    df_sellers                  = pd.read_csv("data_file/business/olist_sellers_dataset.csv")
    df_products                 = pd.read_csv("data_file/business/olist_products_dataset.csv")
    df_product_category_name    = pd.read_csv("data_file/business/product_category_name_translation.csv")

    df_order['order_purchase_timestamp'] = pd.to_datetime(df_order['order_purchase_timestamp'])
    df_order.info()

    # 데이터 확인
    print(df_customers.shape)
    print(df_geolocation.shape)
    print(df_order_items.shape)
    print(df_order_payments.shape)
    print(df_order_reviews.shape)
    print(df_order.shape)
    print(df_sellers.shape)
    print(df_products.shape)
    print(df_product_category_name.shape)

    # 결측값 측정
    # print(df_customers.isnull().sum())
    # print(df_geolocation.isnull().sum())
    # print(df_order_items.isnull().sum())
    # print(df_order_payments.isnull().sum())
    # print(df_order_reviews.isnull().sum())
    # print(df_order.isnull().sum())
    # print(df_sellers.isnull().sum())
    # print(df_products.isnull().sum())
    # print(df_product_category_name.isnull().sum())

    # 전체 데이터 구조확인
    print(df_order.tail())

    # 전체 칼럼 확인
    print(df_order.columns)

    print(df_order.info())


    print(df_order.describe())

    print("\n", "=" * 3, "01.", "=" * 3)
    print('결측치 총 개수 :', df_order.isnull().sum().sum())
    print('1번 칼럼이 빈 값 전체 출력', df_order[df_order.isnull().any(axis=1)])

    sns.heatmap(df_order.isnull(), cbar=False)
    plt.show()
    df_order_null = df_order[df_order.isnull().any(axis=1)]
    df_order_clean = df_order.dropna(axis=0)
    df_order_clean = df_order_clean.reset_index()

    # 해당 칼럼에 있는 종류 보기
    print(df_order_clean['order_status'].unique())
    # 해당 칼럼에 있는 종류의 총 수 보기
    print(df_order_clean['order_status'].value_counts())

    A = df_order_clean[df_order_clean['order_status'] == 'canceled'].shape[0]
    B = df_order_null[df_order_null['order_status'] == 'canceled'].shape[0]

    temp = pd.DataFrame(columns=['del_finished', 'del_not_finished'],
                        index=['cancel_cnt'])
    temp.loc['cancel_cnt', 'del_finished'] = A
    temp.loc['cancel_cnt', 'del_not_finished'] = B
    print(temp)

    temp.T.plot(kind='barh')
    plt.show()

    # object 제외하고 describe 출력
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
    print(df_order_clean.describe(exclude=[np.object]))

    print("\n", "=" * 3, "02.", "=" * 3)
    """
        olist_orders_dataset 테이블에 새로운 정보 추가
        order_purchase_timestamp : 구매 시작 날짜/시간
        order_approved_at : 결제 완료 날짜/시간
        order_delivered_customer_date : 실제 고객한테 배달완료된 날짜/시간
        order_estimated_delivery_date : 시스템에서 고객한테 표시되는 예상배달날짜
        order_approved_at - order_purchase_timestamp : pay_lead_time(단위: 분)
        order_delivered_customer_date - order_approved_at : delivery_lead_time(단위: 일)
        order_estimated_delivery_date - order_delivered_customer_date : estimated_date_miss(단위: 일)
    """
    df_order_clean['pay_lead_time'] = df_order_clean['order_approved_at'] - df_order_clean['order_purchase_timestamp']
    # 분 단위로 변경

    df_order_clean['pay_lead_time_m'] = df_order_clean['pay_lead_time'].astype('timedelta64[m]')
    df_order_clean['pay_lead_time_m']
    print("\n", "=" * 3, "03.", "=" * 3)


business_01()



def business_temp():
    """
        subject
            Machine_Running
        topic
            EX6. 비즈니스 데이터 실습
        content
            01
        Describe

        sub Contents
            01.
    """

    print("\n", "=" * 5, "temp", "=" * 5)
    font_set()
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    df = pd.read_csv("data_file/olist_orders_dataset.csv")

    # 데이터 확인
    print(df.shape)

    # 결측값 측정
    print(df.isnull().sum())

    # 전체 데이터 구조확인
    print(df.tail())

    # 전체 칼럼 확인
    print(df.columns)
    print(df.info())

    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# business_temp()
