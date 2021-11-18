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

import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from util.DBCon import DBConn

def bda_data_analysis():

    print("\n", "=" * 5, "02", "=" * 5)
    df = pd.read_csv("./data_file/BDA_DATA.csv", delimiter=',',  engine='python', error_bad_lines='false')
    print(df.head())
    print(df.shape)

    # 결측값 측정
    print(df.isnull().sum())

    # 전체 데이터 구조확인
    print(df.tail())

    # 전체 칼럼 확인
    print(df.columns)
    print(df.info())




def DBAccess(m_SQL):
    obj_parser = argparse.ArgumentParser(description='This code is written for Preprocessing of Data collection.')
    obj_parser.add_argument('--real_time', type=str, default='N', required=False, choices=['Y', 'N'],
                            help='Choose WHETHER REAL TIME ANALYSIS? (Y/N)')
    args = obj_parser.parse_args()
    V_RL_YN = args.real_time
    return DBConn(V_RL_YN).selectDB(m_SQL)


class select_sql:

    def BOT_EXCEPTURL_SELECT(option):
        print(option)
        m_sql = ""
        if option == "ALL":
            m_sql = """
                    SELECT * FROM bda_colc_categorydata
            """
        return m_sql

    def BOT_COLC_URL_TABLE_INSERT(option, lst):
        print(option)
        m_sql = ""
        if option == "BOT_COLCURL":
            m_sql = """
                    INSERT INTO  BOT_COLCURL (
                          COLC_CNT 	
                        , CHAIN_CNT
                        , URL	
                        , DEGREE 		
                        , USE_YN 		
                        , REGR_ID 		
                        , REGI_DT	
                        , UPDA_ID 	
                        , UPDA_DT                        
                    ) VALUES (
                     ?, ?, ?, ?, ?, 'MSG', SYSDATETIME, 'MSG' SYSDATETIME
                    ) """ % (lst)

        return m_sql

