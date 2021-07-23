"""
    subject
        Machine_Running
    topic
        마케팅 데이터 실습

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

def font_set():
    font_path = "C:\Windows\Fonts\HYGTRE.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font)
    plt.rc('axes', unicode_minus=False)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], label='한글테스트용')
    plt.legend()
    plt.show()


def covid19_01():
    """
        subject
            Machine_Running
        topic
            EX6. covid19_01
        content
            01
        Describe

        sub Contents
            01.
    """
    font_set()

    import warnings
    warnings.filterwarnings("ignore")

    print("\n", "=" * 5, "01", "=" * 5)
    """
        기본 체크 작업
        1. 데이터 확인   : df.shape, df.head(), df.tail()
        2. 결측값 측정   : df.info(), df.isnull().sum()
        3. 분석에 필요한 컬럼만 선택
        4. 기술통계 확인 : df.describe()
        5. 변수간의 correlation 확인
        6. 변수간의 pairplot 확인
        7. Label, Feature(인풋 변수, 독립 변수) 지정
    """
    fig, axes = plt.subplots(2, 2, sharey=False, tight_layout=True, figsize=(15, 6), num='mobile game')

    # 데이터 확인
    df = pd.read_csv("data_file/Covid-19-World-Dataset.csv", engine='python', error_bad_lines='false')
    print(df.shape)

    # 결측값 측정

    print(df.isnull().sum())

    # 전체 데이터 구조확인
    print(df.tail())

    # 전체 칼럼 확인
    print(df.columns)
    print(df.info())

    # 분석에 필요한 컬럼만 선택
    """
         #   Column                  Non-Null Count  Dtype 
        ---  ------                  --------------  ----- 
         0   iso_code                             99144 non-null  object 
         1   continent                            94493 non-null  object 
         2   location                             99144 non-null  object 
         3   date                                 99144 non-null  object 
         4   total_cases                          95579 non-null  float64
         5   new_cases                            95576 non-null  float64
         6   total_deaths                         85461 non-null  float64
         7   new_deaths                           85617 non-null  float64
         8   reproduction_rate                    78887 non-null  float64
         9   icu_patients                         9941 non-null   float64
         10  hosp_patients                        12350 non-null  float64
         11  weekly_icu_admissions                874 non-null    float64
         12  weekly_hosp_admissions               1546 non-null   float64
         13  new_tests                            44501 non-null  float64
         14  total_tests                          44184 non-null  float64
         15  total_tests_per_thousand             44184 non-null  float64
         16  new_tests_per_thousand               44501 non-null  float64
         17  positive_rate                        48297 non-null  float64
         18  tests_per_case                       47699 non-null  float64
         19  tests_units                          53402 non-null  object 
         20  total_vaccinations                   16332 non-null  float64
         21  people_vaccinated                    15531 non-null  float64
         22  people_fully_vaccinated              12678 non-null  float64
         23  new_vaccinations                     13609 non-null  float64
         24  total_vaccinations_per_hundred       16332 non-null  float64
         25  people_vaccinated_per_hundred        15531 non-null  float64
         26  people_fully_vaccinated_per_hundred  12678 non-null  float64
         27  stringency_index                     83470 non-null  float64
         28  population                           98501 non-null  float64
         29  population_density                   92115 non-null  float64
         30  median_age                           88602 non-null  float64
         31  aged_65_older                        87606 non-null  float64
         32  aged_70_older                        88112 non-null  float64
         33  gdp_per_capita                       88944 non-null  float64
         34  extreme_poverty                      60038 non-null  float64
         35  cardiovasc_death_rate                88974 non-null  float64
         36  diabetes_prevalence                  91259 non-null  float64
         37  female_smokers                       69597 non-null  float64
         38  male_smokers                         68580 non-null  float64
         39  handwashing_facilities               44708 non-null  float64
         40  hospital_beds_per_thousand           81057 non-null  float64
         41  life_expectancy                      94145 non-null  float64
         42  human_development_index              89116 non-null  float64
         43  excess_mortality                     3503 non-null   float64
        dtypes: float64(39), object(5)
        memory usage: 33.3+ MB
    """
    # Date 정보 타입 수정
    df['date'] = pd.to_datetime(df['date'])
    df_latest = df[df['date'] == max(df['date'])]

    # 국가별 합계 구하기
    df_conuntry_sum = df_latest.groupby('location')['new_cases', 'new_deaths'].sum().reset_index()
    df_conuntry_sum = df_conuntry_sum.sort_values(by='new_cases', ascending=False).reset_index(drop=True)

    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)



def youtube_temp():
    """
        subject
            Machine_Running
        topic
            EX6. covid19_01
        content
            01
        Describe

        sub Contents
            01.
    """

    print("\n", "=" * 5, "temp", "=" * 5)


    print("\n", "=" * 3, "01.", "=" * 3)
    print("\n", "=" * 3, "02.", "=" * 3)
    print("\n", "=" * 3, "03.", "=" * 3)

# youtube_temp()

def etc_oprtion():
    # 내림차순 정렬
    # df_sorted = df.sort_values(by='views', ascending=False)

    # 중복제거
    # df_sorted.drop_duplicates(['title', 'channel_title'], keep='first')

    # 변수간의 correlation 확인 df.corr() 시각화
    # corr = df.corr()
    # print(corr)
    # annot=True 숫자 표시해줌

    # ax_temp = axes[0, 0]
    # ax_temp.set_title('HeatMap')
    # sns.heatmap(corr, annot=True, ax=ax_temp)
    return

