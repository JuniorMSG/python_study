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
    # plt.show()


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

font_set()
df_cust = pd.read_csv("data_file/business/olist_customers_dataset.csv")
df_order = pd.read_csv("data_file/business/olist_orders_dataset.csv")


# object 타입의 칼럼을 시간 데이터 타입으로 변경해줍니다.
df_order['order_purchase_timestamp'] = pd.to_datetime(df_order['order_purchase_timestamp'])
df_order.info()

df_order = pd.read_csv('data_file/business/olist_orders_dataset.csv',
                       parse_dates=['order_purchase_timestamp',
                                    'order_approved_at',
                                    'order_delivered_carrier_date',
                                    'order_delivered_customer_date',
                                    'order_estimated_delivery_date'
                                    ])

# 데이터 확인
print(df_order.shape)

# 결측값 측정
print(df_order.isnull().sum())

# 전체 데이터 구조확인
print(df_order.tail())

# 전체 칼럼 확인
print(df_order.columns)
print(df_order.info())

# 요약 확인
print(df_order.describe())

print('결측치 총 개수 :', df_order.isnull().sum().sum())
print('1번 칼럼이 빈 값 전체 출력', df_order[df_order.isnull().any(axis=1)])


sns.heatmap(df_order.isnull(), cbar=False)
# plt.show()
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
# plt.show()

# object 제외하고 describe 출력
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
# print(df_order_clean.describe(exclude=[np.object]))

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

df_order_clean['delivery_lead_time'] = df_order_clean['order_delivered_customer_date'] - df_order_clean['order_approved_at']
df_order_clean['delivery_lead_time_D'] = df_order_clean['delivery_lead_time'].astype('timedelta64[D]')
df_order_clean['delivery_lead_time_D']

df_order_clean['estimated_date_miss'] = df_order_clean['order_estimated_delivery_date'] - df_order_clean['order_delivered_customer_date']
df_order_clean['estimated_date_miss_D'] = df_order_clean['estimated_date_miss'].astype('timedelta64[D]')
df_order_clean['estimated_date_miss_D']

df_order_clean['pay_lead_time_m'] = df_order_clean['pay_lead_time_m'].astype(int)
df_order_clean['delivery_lead_time_D'] = df_order_clean['delivery_lead_time_D'].astype(int)
df_order_clean['estimated_date_miss_D'] = df_order_clean['estimated_date_miss_D'].astype(int)

print(df_order_clean.sample(5))

fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='mobile game')
ax_temp = axes[0, 0]
ax_temp.set_title('pay_lead_time_m')
sns.distplot(df_order_clean['pay_lead_time_m'], ax=ax_temp)

ax_temp = axes[0, 1]
ax_temp.set_title('delivery_lead_time_D')
sns.distplot(df_order_clean['delivery_lead_time_D'], ax=ax_temp)

ax_temp = axes[1, 0]
ax_temp.set_title('estimated_date_miss_D')
sns.distplot(df_order_clean['estimated_date_miss_D'], ax=ax_temp)
# plt.show()

# 새로 추가한 칼럼들의 요약 정보

print(df_order_clean[['pay_lead_time_m', 'delivery_lead_time_D', 'estimated_date_miss_D']].describe())

fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='mobile game')
ax_temp = axes[0, 0]
ax_temp.set_title('pay_lead_time_m')
sns.boxplot(data=df_order_clean['pay_lead_time_m'], ax=ax_temp)

ax_temp = axes[0, 1]
ax_temp.set_title('delivery_lead_time_D')
sns.boxplot(data=df_order_clean['delivery_lead_time_D'], ax=ax_temp)

ax_temp = axes[1, 0]
ax_temp.set_title('estimated_date_miss_D')
sns.boxplot(data=df_order_clean['estimated_date_miss_D'], ax=ax_temp)
# plt.show()

#이상한 데이터 확인
# df_order_clean[df_order_clean['pay_lead_time_m']==44486]
# df_order_clean[df_order_clean['delivery_lead_time_D']==208]
# df_order_clean[df_order_clean['estimated_date_miss_D']==146]
print(df_order_clean[df_order_clean['delivery_lead_time_D'] <= 0])

df_order_time = df_order_clean[['pay_lead_time_m', 'delivery_lead_time_D', 'estimated_date_miss_D']]
print(df_order_time.shape)

def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)

    return np.where((data > upper_bound) | (data < lower_bound))

# pay_lead_time, delivery_lead_time, estimated_date_miss 이상치 수
print('pay_lead_time_m :', outliers_iqr(df_order_time['pay_lead_time_m'])[0].shape[0])
print('delivery_lead_time_D :', outliers_iqr(df_order_time['delivery_lead_time_D'])[0].shape[0])
print('estimated_date_miss : ', outliers_iqr(df_order_time['estimated_date_miss_D'])[0].shape[0])

# 세 칼럼의 이상치 row 인덱스 출력
pay_lead_outlier_index = outliers_iqr(df_order_time['pay_lead_time_m'])[0]
del_lead_outlier_index = outliers_iqr(df_order_time['delivery_lead_time_D'])[0]
est_lead_outlier_index = outliers_iqr(df_order_time['estimated_date_miss_D'])[0]

# 이상치 row들의 인덱스는 넘파이 배열로 출력됩니다.
print(type(pay_lead_outlier_index), pay_lead_outlier_index)

# pay_lead_time 이상치에 해당되는 값 출력
print('pay_lead_outlier_index', df_order_time.loc[pay_lead_outlier_index, 'pay_lead_time_m'])
print('del_lead_outlier_index', df_order_time.loc[del_lead_outlier_index, 'delivery_lead_time_D'])
print('est_lead_outlier_index', df_order_time.loc[est_lead_outlier_index, 'estimated_date_miss_D'])

# numpy concat을 통한 array 배열 합치기
lead_outlier_index = np.concatenate((pay_lead_outlier_index,
                                     del_lead_outlier_index,
                                     est_lead_outlier_index
                                     ), axis=None)
print(len(lead_outlier_index))
print(lead_outlier_index.shape)

# for문을 이용해 이상치가 아닌 리드타임 값의 인덱스를 추려줍니다. 20개까지만 출력
lead_not_outlier_index = []
for i in df_order_time.index:
    if i not in lead_outlier_index:
        lead_not_outlier_index.append(i)

df_order_time_clean = df_order_time.loc[lead_not_outlier_index]
df_order_time_clean = df_order_time_clean.reset_index(drop=True)
print(df_order_time_clean.shape)
print(df_order_time_clean.info())

# 클렌징한 DF 의 요약정보
print(df_order_time_clean.describe())
fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='clean box_plot')
ax_temp = axes[0, 0]
ax_temp.set_title('pay_lead_time_m')
sns.boxplot(data=df_order_time_clean['pay_lead_time_m'], ax=ax_temp)

ax_temp = axes[0, 1]
ax_temp.set_title('delivery_lead_time_D')
sns.boxplot(data=df_order_time_clean['delivery_lead_time_D'], ax=ax_temp)

ax_temp = axes[1, 0]
ax_temp.set_title('estimated_date_miss_D')
sns.boxplot(data=df_order_time_clean['estimated_date_miss_D'], ax=ax_temp)
# plt.show()



font_set()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
df_cust = pd.read_csv("data_file/business/olist_customers_dataset.csv")
df_order_items = pd.read_csv("data_file/business/olist_order_items_dataset.csv")

# 데이터 확인
print(df_cust.shape)

# 결측값 측정
print(df_cust.isnull().sum())

# 전체 데이터 구조확인
print(df_cust.tail())
print(df_cust.sample(3))

# 전체 칼럼 확인
print(df_cust.columns)
print(df_cust.info())

# states 별 고객 수
cust_stat = pd.DataFrame(df_cust['customer_state'].value_counts()).reset_index()
cust_stat.columns = ['states', 'people_lives']
print(cust_stat)

sns.barplot(x='states', y='people_lives', data=cust_stat)
# plt.show()


# 도시별 고객이 살고 있는 비율
df_customer_city = pd.DataFrame(df_cust['customer_city'].value_counts(normalize=True) * 100).reset_index()
df_customer_city.columns = ['city', 'people_lives_perc']
print(df_customer_city.loc[:10, :])

# pie chart
# 거주비율 상위 10개만

# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.pie.html
# https://medium.com/@kvnamipara/a-better-visualisation-of-pie-charts-by-matplotlib-935b7667d77f

labels = df_customer_city['city'].values[:10]
sizes = df_customer_city['people_lives_perc'].values[:10]

explode = (0.1, 0.1, 0, 0, 0,
           0, 0, 0, 0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,
        explode=explode,
        autopct='%1.1f%%',
        shadow=True, startangle=30,
        textprops={'fontsize': 16})

ax1.axis('equal')

plt.tight_layout()
plt.title('도시별 고객이 살고 있는 비율', fontsize=20)
# plt.show()

# plt.pie(df_customer_city['people_lives_perc'],labels=df_customer_city['city'], autopct='%.0f%%', shadow=True)


font_set()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
df_order_item = pd.read_csv("data_file/business/olist_order_items_dataset.csv")

# 데이터 확인
print(df_order_item.shape)

# 결측값 측정
print(df_order_item.isnull().sum())

# 전체 데이터 구조확인
print(df_order_item.tail())
print(df_order_item.sample(3))

# 전체 칼럼 확인
print(df_order_item.columns)
print(df_order_item.info())

#  일단, item_id를 제일 많이 갖고 있는 order_id를 출력해봅니다.
temp = pd.DataFrame(df_order_item.groupby(by=['order_id'])['order_item_id'].count().reset_index())
temp.columns = ['order_id', 'order_item 수']
temp[temp['order_item 수'] == temp['order_item 수'].max()]

# order item 숫자가 가장 큰 order_id를 통해 데이터를 확인해봅니다.
df_order_item[df_order_item['order_id'] == '8272b63d03f5f79c56e9e4120aec44ef']

# 위 결과로 알 수 있는 점!
# order_item_id 칼럼의 뜻은 하나의 주문(order_id)에서 구매한 상품들의 수를 뜻합니다. (상품 종류에 상관없이)
# 위 결과의 order_item_id 1번인 상품과 2번, 12번 상품의 price는 1.2로 같지만, 21번 상품은 7.8로 다른 것을 볼 때, price 칼럼은 "상품 단가"를 의미한다고 생각해볼 수 있습니다.
# (위의 결과에서 총 상품 종류는 3가지)
# 하나의 주문번호에서 "상품별 매출액"을 산출해내기 위해서는 각 상품들의 "구매수량" 칼럼을 추가해주어야 합니다. (상품단가 * 구매수량 = 상품별 매출액)

# 주문한 상품 수량 : order_prod_quantity
df_qt = pd.DataFrame(df_order_item.groupby(by=['order_id', 'product_id'])['order_item_id'].count().reset_index())

# 칼럼명 변경
df_qt.columns = ['order_id', 'product_id', 'order_prod_quantity']

print(df_qt.sort_values('order_prod_quantity', ascending=False))


# 원하는 값으로 들어갔는지 예제로 확인해봅니다.
df_qt[df_qt['order_id'] == '8272b63d03f5f79c56e9e4120aec44ef']



# 상품 별 주문수량을 추가해주기 위한 merge

# 사용할 order_item 칼럼
df_order_item_col = ['order_id', 'product_id', 'seller_id',
                     'shipping_limit_date', 'price', 'freight_value']

# merge
df_order_item = pd.merge(df_order_item[df_order_item_col], df_qt, how='inner', on=['order_id', 'product_id'])

# 칼럼 순서 재배치
df_order_item = df_order_item[['order_id', 'product_id', 'price', 'freight_value', \
                               'order_prod_quantity', 'shipping_limit_date', 'seller_id']]

print(df_order_item)

# 제대로 결합 되었는지 예제로 확인해봅니다.
# 예제 order_id : 8272b63d03f5f79c56e9e4120aec44ef
print(df_order_item[df_order_item['order_id'] == '8272b63d03f5f79c56e9e4120aec44ef'])

# 발생된 중복 row를 제거해줍니다.
df_order_item.drop_duplicates(inplace=True)
df_order_item.reset_index(drop=True, inplace=True)
print(df_order_item)

# 주문한 상품 별 매출액
df_order_item['order_amount'] = df_order_item['price'] * df_order_item['order_prod_quantity']
df_order_item.reset_index(drop=True, inplace=True)
print(df_order_item)

print(df_order_item[df_order_item['order_id']=='8272b63d03f5f79c56e9e4120aec44ef'])


df_product = pd.read_csv("data_file/business/olist_products_dataset.csv")
df_category = pd.read_csv("data_file/business/product_category_name_translation.csv")
# df_order_items = pd.read_csv("data_file/business/olist_order_items_dataset.csv")

# 데이터 확인
print(df_product.shape)

# 결측값 측정
print(df_product.isnull().sum())

# 전체 데이터 구조확인
print(df_product.tail())
print(df_product.sample(3))

# 전체 칼럼 확인
print(df_product.columns)
print(df_product.info())

# df_category 와 영어명 매칭

df_product_cat = pd.merge(df_product, df_category, \
                          how='left', on=['product_category_name'])

# 칼럼 순서 재배치
df_product_cat = df_product_cat[['product_id', 'product_category_name', 'product_category_name_english',
                                 'product_name_lenght', 'product_description_lenght',
                                 'product_photos_qty', 'product_weight_g',
                                 'product_length_cm', 'product_height_cm', 'product_width_cm']]

print(df_product_cat)
print(df_product_cat.isnull().sum())

# df_order_item을 기준으로 merge
df_order_item_prod = pd.merge(df_order_item, df_product_cat, how='left', on=['product_id'])
print(df_order_item_prod.tail())
print(df_order_item_prod.describe())

# 결측치 확인
print(df_order_item_prod.isnull().sum())
print(df_order_item_prod[df_order_item_prod.isnull().any(axis=1)])

df_order_item_prod_clean = df_order_item_prod.dropna(axis=0)  # axis=1이면 칼럼을 기준으로
df_order_item_prod_clean.reset_index(drop=True, inplace=True)


# 결측치 처리를 해준 데이터프레임
print(df_order_item_prod_clean.tail())
print(df_order_item_prod_clean.shape)
print(df_order_item_prod_clean.isnull().sum())

# pandas 소수점 출력 설정 : 소수점 2째자리까지만
pd.options.display.float_format = '{:.2f}'.format

# 상품카테고리 별 주문수 확인
print(df_order_item_prod_clean['product_category_name_english'].value_counts())

# 상품 카테고리 중 주문수가 상위 10개
cat_top10 = df_order_item_prod_clean['product_category_name_english'].value_counts()[:10]
print(cat_top10)

# 상품 카테고리가 cat_top10 의 값에 포함되는 row만 출력하기

df_cat_10 = df_order_item_prod_clean[df_order_item_prod_clean['product_category_name_english'].isin(cat_top10.index)].reset_index(drop=True)
print(df_cat_10.tail())

# 상위 10개의 카테고리별 정보 - 상품등록정보
cat_info1_col = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty']
df_cat_10.groupby('product_category_name_english')[cat_info1_col].mean()
print(df_cat_10.describe())

# 카테고리별 정보2 확인
cat_info2_col = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
df_cat_10.groupby('product_category_name_english')[cat_info2_col].mean()
print(df_cat_10.describe())

# 상품 카테고리 종류 수
print("상품 카테고리 종류 수 : {} 종류".format(len(df_order_item_prod_clean['product_category_name_english'].unique())))

# 매출액 기준 상품 카테고리
temp = pd.DataFrame(df_order_item_prod_clean.groupby(by=['product_category_name_english'])['order_amount'].sum())

# 매출액 높은 순으로 정렬
temp = temp.sort_values(by='order_amount', ascending=False)
print(temp)

import squarify
plt.figure(figsize=(12, 10))
squarify.plot(sizes=temp['order_amount'][:10],
              label=temp.index.values[:10], alpha=.7)

# 카테고리별 주문수 확인
df_cat_order_cnt = pd.DataFrame(df_order_item_prod_clean['product_category_name_english'].value_counts())
df_cat_order_cnt = df_cat_order_cnt.reset_index()
df_cat_order_cnt.columns = ['category', 'order_cnt']
print(df_cat_order_cnt)

# 카테고리별 매출액순
df_cat_amount = pd.DataFrame(df_order_item_prod_clean.groupby(by=['product_category_name_english'])['order_amount'].sum())
df_cat_amount = df_cat_amount.sort_values(by='order_amount', ascending=False)
df_cat_amount = df_cat_amount.reset_index()
df_cat_amount.columns = ['category', 'order_amount']
print(df_cat_amount)

# 카테고리별 주문 비율
df_cat_order_cnt['order_cnt_perc'] = (df_cat_order_cnt['order_cnt'] / sum(df_cat_order_cnt['order_cnt'])) * 100
print(df_cat_order_cnt)

# 카테고리별 매출 비율
df_cat_amount['order_amount_perc'] = (df_cat_amount['order_amount'] / sum(df_cat_amount['order_amount'])) * 100
print(df_cat_amount)

# 카테고리별 주문수와 매출액 테이블 결합
df_cat = pd.merge(df_cat_order_cnt, df_cat_amount, how='inner', on='category')
print(df_cat)
df_cat = df_cat.sort_values(by='order_amount', ascending=False)
df_cat = df_cat.reset_index(drop=True)
print(df_cat)

# melt
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html
# 매출액 기준 상위 10개 카테고리만 출력
df_cat_melt = pd.melt(df_cat[:10],
                      id_vars=['category'],
                      value_vars=['order_cnt_perc', 'order_amount_perc'])
print(df_cat_melt)

# barplot으로 시각화
plt.figure(figsize=(12, 10))
ax = sns.barplot(data=df_cat_melt,
                 x="category",
                 y="value",
                 hue="variable",
                 color="salmon"
                 )
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
# plt.show()

"""
    카테고리에 따른 월별 매출액이 어떠한지 확인해봅니다.
    현재 df_order_item_prod_clean 테이블에는 구매가 발생한 상세월에 대한 정보가 없습니다. 따라서, 이러한 정보를 포함하고 있는 order_df 테이블과 함께 추출하여야 합니다.
    
    집계 기준
    
    매출이 발생한 시점은 "order_approved_at"을 기준으로 합니다.
    "order_status"가 "delivered"된 값만 집계합니다.
"""
# 활용테이블의 shape(행, 열) 수 확인
print(df_order_clean.shape)
print(df_order_item_prod_clean.shape)

# apply lambda를 이용해서 날짜데이터를 string으로 변환후 원하는 포맷으로 출력하기
df_order_clean['order_date'] = df_order_clean['order_approved_at'].apply(lambda x : x.strftime('%Y%m') )
print(df_order_clean['order_date'])
print(df_order_clean.head())

# 두 테이블을 merge 합니다.

df_order_tmp = pd.merge(df_order_clean, df_order_item_prod_clean, how='inner', on=['order_id'])
print(df_order_tmp.sample(3))

# 결측치 확인
print(df_order_tmp.isnull().sum())
print(df_order_tmp.tail())

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html
df_order_pivot = df_order_tmp.pivot_table(
                              values='order_amount',
                              index='product_category_name_english',
                              columns='order_date',
                              aggfunc='mean')

print(df_order_pivot.tail())

# health_beauty 카테고리의 연월별 평균 매출액 출력
print(df_order_pivot.loc["health_beauty",:])

# 연월별 평균매출액 출력
# null값은 제외
df_health_beauty = pd.DataFrame(df_order_pivot.loc["health_beauty",:])
df_health_beauty = df_health_beauty.reset_index()
df_health_beauty.columns = ['date', 'health_beauty_amount']
df_health_beauty.dropna(inplace=True)
print(df_health_beauty)

# 막대그래프로 시각화
plt.figure(figsize=(12,8))
sns.barplot(data = df_health_beauty,
                 x='date',
                 y="health_beauty_amount",
                 palette="Blues_d"
                )
# ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.xticks(fontsize=14, rotation=30)
# plt.show()

# lineplot으로 시각화

plt.figure(figsize=(12,8))
ax = sns.lineplot(data = df_health_beauty,
                 x='date',
                 y="health_beauty_amount",
                 palette="Blues_d"
                )
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
# plt.show()


# olist_order_payments 테이블
# 고객들이 주문 결제를 어떻게 했는지, 결제 정보가 담긴 테이블
df_order_pay = pd.read_csv("data_file/business/olist_order_payments_dataset.csv")

# 데이터 확인
print(df_order_pay.shape)

# 결측값 측정
print(df_order_pay.isnull().sum())

# 전체 데이터 구조확인
print(df_order_pay.tail())
print(df_order_pay.sample(3))

# 전체 칼럼 확인
print(df_order_pay.columns)
print(df_order_pay.info())

# 요약정보 출력
print(df_order_pay.describe())

# payment_sequential이 최대값인 order_id 확인
df_order_pay[df_order_pay['order_id']=='fa65dad1b0e818e3ccc5cb0e39231352'].sort_values(by='payment_sequential')

# 데이터스키마에는 payment_sequential 칼럼은 "지불 방법의 종류"라고 설명되어 있습니다.
# 위의 결과를 보면 동일한 주문번호 내 payment_type이 voucher로, payment_sequential이 다르게 부여된 것으로 볼 때,
# 이 주문은 각기 다른 바우처(=상품권)를 사용된 것으로 예상해볼 수 있습니다.

print(df_order_pay[df_order_pay['payment_sequential']==df_order_pay['payment_sequential'].max()])


# credit_card', 'boleto', 'voucher', 'debit_card', 'not_defined'
print(df_order_pay['payment_type'].unique())

# 결제방법이 신용카드인 경우 확인
df_credit = df_order_pay[df_order_pay['payment_type']=='credit_card']
print(df_credit.tail())
print(df_credit.describe())

# 결제 방법 별 payment_installments 평균값 확인
print(df_order_pay.groupby(['payment_type'])['payment_installments'].mean())

# 고객들이 많이 선택한 결제 방법과 그 수
print(df_order_pay['payment_type'].value_counts())

# 고객들이 많이 선택한 결제 방법 % 비율
print(df_order_pay['payment_type'].value_counts(normalize=True)*100)

# pie 그래프

temp = pd.DataFrame(df_order_pay['payment_type'].value_counts(normalize=True)*100)
temp = temp[temp.index != 'not_defined'] # 너무 작은 수라 제외함

labels = temp.index
sizes = temp['payment_type']

explode = (0.1, 0, 0, 0.1)

plt.pie(
        sizes,
        labels=labels,
        explode=explode,
        autopct='%1.2f%%', # second decimal place
        shadow=True,
        startangle=50,
        textprops={'fontsize': 14} # text font size
        )

plt.axis('equal') #  equal length of X and Y axis
plt.title('결제방식의 종류별 비율', fontsize=20)
# plt.show()


df_order_review = pd.read_csv('data_file/business/olist_order_reviews_dataset.csv',
                             parse_dates=['review_creation_date','review_answer_timestamp'])

# 데이터 확인
print(df_order_review.shape)

# 결측값 측정
print(df_order_review.isnull().sum())

# 전체 데이터 구조확인
print(df_order_review.tail())
print(df_order_review.sample(3))

# 전체 칼럼 확인
print(df_order_review.columns)
print(df_order_review.info())

# 요약정보 출력
print(df_order_review.describe())

# 리뷰 점수 별 수


df_review = pd.DataFrame(df_order_review['review_score'].value_counts())
df_review.reset_index(inplace=True)
df_review.columns = ['review_score', 'cnt']
print(df_review)

# 막대그래프 시각화
sns.barplot(data = df_review, x='review_score', y='cnt')

# review_score 별 비율
print(df_order_review['review_score'].value_counts(normalize=True)*100)

# review_score 별 비율 시각화 : piechart
temp = pd.DataFrame(df_order_review['review_score'].value_counts(normalize=True)*100)
labels = temp.index
sizes = temp['review_score']

explode = (0.1, 0, 0, 0, 0)

plt.pie(
        sizes,
        labels=labels,
        explode=explode,
        autopct='%1.2f%%', # second decimal place
        shadow=True,
        startangle=70,
        textprops={'fontsize': 14} # text font size
        )

plt.axis('equal') #  equal length of X and Y axis
plt.title('리뷰 점수별 분포 비율', fontsize=20)
# plt.show()

# 고객의 리뷰 작성까지 걸리는 시간
df_order_review['answer_lead_time'] = df_order_review['review_answer_timestamp'] - df_order_review['review_creation_date']
print(df_order_review['answer_lead_time'])

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.total_seconds.html

# 고객의 리뷰 작성까지 걸리는 시간을 초 단위로 변환하기
# df_order_review['answer_lead_time'][0].components
df_order_review['answer_lead_time'][0].total_seconds()

# 고객의 리뷰 작성까지 걸리는 시간을 초 단위로 변환하기
df_order_review['answer_lead_time_seconds'] = df_order_review['answer_lead_time'].apply(lambda x : x.total_seconds())
print(df_order_review['answer_lead_time_seconds'])

# answer_lead_time_seconds 히스토그램
plt.figure(figsize=(12,6))
sns.distplot(df_order_review['answer_lead_time_seconds'])

# answer_lead_time_seconds 이상치 확인
plt.figure(figsize=(12,8))
sns.boxplot(data=df_order_review['answer_lead_time_seconds'], color='yellow')

# answer_lead_time_seconds 이상치 수
print("이상치 수 : {} 건".format(outliers_iqr(df_order_review['answer_lead_time_seconds'])[0].shape[0]))

# # answer_lead_time_seconds 이상치 출력
df_order_review.loc[outliers_iqr(df_order_review['answer_lead_time_seconds'])[0],'answer_lead_time_seconds']

# answer_lead_time 이상치 출력
df_order_review.loc[outliers_iqr(df_order_review['answer_lead_time'])[0],'answer_lead_time'].sort_values(ascending=False)


"""
--- 06

"""
# 이번에는 고객의 만족도와 관계있는 칼럼들을 살펴보겠습니다.
# 활용테이블
# olist_orders_dataset (= df_order_clean)
# olist_order_reviews_dataset (=df_order_review)

# df_order_clean을 기준으로 merge
df_satisfy = pd.merge(df_order_clean, df_order_review, how='left', on=['order_id'])
print(df_satisfy.head())

# 요약 정보 확인\
# 총 23개의 칼럼
print(df_satisfy.info())

#결측치
print(df_satisfy.isnull().sum())

# columns 확인
print(df_satisfy.columns)

# 고객 만족도(review_score) 확인을 위한 데이터셋 생성
df_cust_sf = df_satisfy[[
                    'pay_lead_time_m',
                    'delivery_lead_time_D',
                    'estimated_date_miss_D',
                    'answer_lead_time_seconds',
                    'review_score'
                    ]]
print(df_cust_sf.sample(3))

# 요약 정보 확인
print(df_cust_sf.describe())

# https://ko.wikipedia.org/wiki/%EC%83%81%EA%B4%80_%EB%B6%84%EC%84%9D
# 상관 분석 corr

# method='pearson' 안쓸경우  ValueError: Must pass 2-d input. shape=() 에러 발생
corr = df_cust_sf.corr(method='pearson')
print(corr)
plt.figure(figsize=(15,10))
sns.heatmap(data = corr,
            annot=True,
            fmt = '.2f',
            linewidths=.5,
            cmap='Blues'
           )

# => 상관관계 확인 결과, "review_score" 는 "delivery_lead_time" 과는 음의 상관관계를,
# "estimated_date_miss" 와는 약한 양의 상관관계를, "answer_lead_time_seconds" 와는 상관관계가 거의 없는 것으로 나타났습니다.
# 즉, 고객한테 배송이 걸리는 시간이 짧을수록, 예정된 날짜보다 빨리 배송될수록 고객의 만족도가 높아짐을 확인할 수 있습니다.

# delivery_lead_time과 estimated_date_miss 간 상관관계를 산점도로 그려봅니다.
plt.figure(figsize=(12,10))
sns.scatterplot(x='delivery_lead_time_D', y='estimated_date_miss_D', data=df_cust_sf)
plt.show()

"""
    "delivery_lead_time"과 "estimated_date_miss" 간에는 강한 음의 상관관계를 확인할 수 있는데, 
    이건 고객한테 배송이 되는 날짜가 짧을(빠를) 경우, 시스템에서 예측한 소요시간과의 차이가 많이남을 의미합니다.
"""

"""
    여섯번째로 살펴볼 테이블 : olist_geolocation_dataset
    지리정보 데이터셋
"""
df_geo = pd.read_csv("data_file/business/olist_geolocation_dataset.csv")

# 지리정보 데이터셋 info
print(df_geo.info())
print(df_geo.shape)

# 지리정보 데이터 확인
print(df_geo.sample(5))

# state 별 도시 수 출력
pd.DataFrame(df_geo.groupby(by=['geolocation_state'])['geolocation_city'].count().sort_values(ascending=False))

# 위도와 경도 정보로 산점도 시각화
df_geo.plot.scatter(x='geolocation_lng', y='geolocation_lat', figsize=(12,8),grid=True)

# state별 색상을 구분하여 산점도로 시각화
# row가 많아 시간이 조금 걸릴 수 있습니다.
# plt.figure(figsize=(14, 10))
# ax = sns.scatterplot(data=df_geo,
#                      x='geolocation_lng',
#                      y='geolocation_lat',
#                      hue='geolocation_state')
# plt.setp(ax.get_legend().get_texts(), fontsize='10')
# plt.show()

df_seller = pd.read_csv("data_file/business/olist_sellers_dataset.csv")

# info 확인
print(df_seller.info())
print(pd.DataFrame(df_seller.groupby(by=['seller_state'])['seller_city'].count().sort_values(ascending=False)))

# seller의 state 별 도시 수
print(pd.DataFrame(df_seller.groupby(by=['seller_state'])['seller_city'].count().sort_values(ascending=False)))

# 도시 별셀러의 수
print(df_seller['seller_city'].value_counts())

# 도시별 고객 수
print(df_cust['customer_city'].value_counts())

# 도시별 고객과 판매자 비율 비교
# df_seller['seller_city']
# df_cust['customer_city']

# seller 데이터
df_seller_lives = pd.DataFrame(df_seller['seller_city'].value_counts(normalize=True)*100)
df_seller_lives.reset_index(inplace=True)
df_seller_lives.columns = ['city', 'seller_lives']

# customer 데이터
df_cust_lives = pd.DataFrame(df_cust['customer_city'].value_counts(normalize=True)*100)
df_cust_lives.reset_index(inplace=True)
df_cust_lives.columns = ['city', 'customer_lives']

# merge
df_seller_cust_lives = pd.merge(df_seller_lives, df_cust_lives,\
                        how='inner', on=['city'])

# 고객 수가 많은 순서대로 출력
df_seller_cust_lives = df_seller_cust_lives.sort_values(by='customer_lives', ascending=False)
df_seller_cust_lives = df_seller_cust_lives.reset_index(drop=True)
print(df_seller_cust_lives)

# 고객 수가 많은 순서의 도시들 상위 10개만 출력
top10 = df_seller_cust_lives[:10]

top10 = pd.melt(top10, id_vars=['city'], value_vars=['seller_lives', 'customer_lives'])

# 고객 수가 많은 순서의 도시들 상위 10개의 고객과 판매자 비율 비교

plt.figure(figsize=(12,10))
ax = sns.barplot(data = top10,
                 x="city",
                 y="value",
                 hue="variable",
#                  color="salmon",
                 palette="Blues_d"
                )
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.show()
"""
    판매자와 고객 모두 상파울루에 제일 많이 거주하고 있지만, 판매자 비율이 상대적으로 더 높음을 알 수 있습니다. 
    반면, 리우데자네이루의 경우, 판매자보다 고객의 비율이 더 높은데, 
    이런 점을 볼 때, 상파울루가 다른 도시들에 비해 판매자 비율이 상대적으로 더 높은 이유를 조금 더 찾아보는 것도 의미가 있겠습니다.
"""


# 판매자 위치 정보 표시하기

# column 이름 변경
df_seller_temp = df_seller.copy()
df_seller_temp.columns = ['seller_id', 'geolocation_zip_code_prefix', 'seller_city', 'seller_state']

# merge
df_seller_geo = pd.merge(df_seller_temp, df_geo, how='left', on=['geolocation_zip_code_prefix'])
df_seller_geo.sample(5)

# seller의 위경도 정보로 산점도 시각화하기
# state별 색상 구분
plt.figure(figsize=(14, 10))
ax = sns.scatterplot(data=df_seller_geo,
                     x='geolocation_lng',
                     y='geolocation_lat',
                     hue='geolocation_state')
plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.show()