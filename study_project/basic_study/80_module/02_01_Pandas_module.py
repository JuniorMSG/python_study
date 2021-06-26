"""
    subject
        Data analysis module

    topic
        Pandas 모듈

    Describe
        데이터 분석을 위한 패키지
            - 엑셀로 할 수 있는 모든 것
            - excel, db, pdf파일
            - 크롤링 (웹 정보 수집)
            - Database 핸들링
            - 시각화
        등이 가능함

    Contents
        01. 기본 사용법 & 용어정리
        02. 파일 읽기
        03. 기본 정보 알아보는 함수
        04. 데이터선택
        05. Null 비어있는 값
        06. 복사, 추가, 삭제
        07. 통계값 다루기 
        08. 피벗 테이블
        09. GroupBy
        10. Nan 값 처리
        11. 삭제(drop), 합치기(concat), 병합(merge)
        12. merge (병합)
        13. Series (열)
        14. apply 
        15. 산술연산 
        16. select_dtypes
        17. 원핫인코딩 (One-hot-encoding)

"""

"""
    01. 기본 사용법 & 용어정리
        01_01. import pandas as pd 
            별칭은 주로 pd를 사용한다.

        01_02. Series (column)
            Series는 어떤 데이터 타입이든 보유할 수 있는 label링 된 1차원 배열이다.
            - 정수, 문자열, float, 객체, 기타 등등을 포함한다.
            - 정수, label을 기반으로하는 인덱싱을 지원한다.
            - 인덱스를 포함한 연산에 사용 할 수 있는 다양한 메서드(method)를 지원한다.
            - 축 레이블은 인덱스와 같이 묶여서 참조될 수 있다.
            
            1차원으로 이루어진 데이터 배열

        01_03. DataFrame 
            2차원으로 이루어진 데이터 배열
"""
from datetime import datetime
from os import chdir


def pandas_01():
    import pandas as pd
    print("\n", "=" * 5, "01. 기본 사용법 & 용어정리", "=" * 5)

    print("\n", "=" * 3, "01_01. import pandas as pd ", "=" * 3)
    print(pd)

    print("\n", "=" * 3, "01_02. Series", "=" * 3)
    series_data = [1,2,3,4]
    series_data = pd.Series(series_data)
    print(series_data)
    print('type(series_data) :', type(series_data))

    print("\n", "=" * 3, "01_03. DataFrame", "=" * 3)

    # 2차원 리스트로 만들기 
    company =[['삼성', 80000, '반도체'] , ['카카오', 120000, 'SNS플랫폼'], ['LGU+', 15000, '통신사']]
    company_df = pd.DataFrame(company)

    # 제목 만들기   
    company_df.columns = ['기업명', '주가', '주력산업']
    print('2차원 리스트 :', company_df)

    # dict로 만들기
    company = {'기업명'     : ['삼성', '카카오', 'LGU+'],
                '주가'      : [80000, 120000, 15000],
                '주력산업'  : ['반도체', 'SNS플랫폼', '통신사'],
    }
    company_df = pd.DataFrame(company)
    print('dict :', company_df)
    print('type(company_df["주가"]) :', type(company_df['주가']))

# pandas_01()

def pandas_02():
    """
        02. 파일 읽기
            02_01. CSV 파일
                CSV 파일이란 
                Comma Separated Value의 약어로써, 쉼표로 구분된 파일을 뜻한다.
                엑셀보다 가볍다. 
                공공데이터 포털에서 제공하는 포맷중 하나이다. 

            02_02. Excel 파일
    """
    print("\n", "=" * 5, "02. 파일 읽기 ", "=" * 5)
    print("\n", "=" * 3, "02_01. CSV 파일 ", "=" * 3)

    # https://www.data.go.kr/data/15077858/fileData.do
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/data.csv', delimiter='\t', encoding='CP949')

    print(data)

    print("\n", "=" * 3, "02_02. Excel 파일 ", "=" * 3)

    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    """
        발생에러 : ImportError: Missing optional dependency 'xlrd'. Install xlrd >= 1.0.0 for Excel support Use pip or conda to install xlrd.
        pip install xlrd - 실패
        xlrd가 업데이트 되면서 xlsx 지원을 하지 않는다고 합니다.
        
        pandas 기본 엔진이 xlrd라서 그렇다는 군요 그래서 엔진을 변경하면
        pip install openpyxl 
    """
    # pip install openpyxl
    data = pd.read_excel(path +'/file_data/company.xlsx', engine='openpyxl')
    print(data.head())

pandas_02()

def pandas_03():

    """
        03. 기본 정보 알아보는 함수
            03_01. index
            03_02. 정보 조회 :
                info()      : 전체 정보를 보여준다.
                describe()  : 산술 연산이 가능한 타입만 출력된다.
                shape       : shape 정보를 알려준다 (행, 열)
            03_03. 요약 출력 :
                head()      : 상위 5개 row 출력
                head(n)     : 상위 n개 row 출력
                tail()      : 하위 5개 row 출력
                tail(n)     : 하위 n개 row 출력
            03_04. 정렬 :
                option = ascending= (Default = True, False 내림차순 정렬)
                sort_index()    : 인덱스 기준 정렬

                option = by='colums'
                sort_values()   : 값 기준 정렬


    """
    print("\n", "=" * 5, "03. ", "=" * 5)

    import pandas as pd
    import os

    
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/data.csv', delimiter=',', encoding='CP949')
    print(data)

    print("\n", "=" * 3, "03_01. index", "=" * 3)
    print(data.columns)
    new_col = ['index', '사례수', '증가', '감소', '변화 없음']
    data.columns = new_col
    print('==컬럼정보==\n', data.columns)
    print('==인덱스정보==\n', data.index)

    print("\n", "=" * 3, "03_02. 정보 조회", "=" * 3)
    print('==info()==       :', data.info())
    print('==describe()==   :', data.describe())
    print('==shape==        :', data.shape)

    print("\n", "=" * 3, "03_03. 요약 출력", "=" * 3)
    print(data.head())
    print(data.head(3))
    print(data.tail())
    print(data.head(3))

    print("\n", "=" * 3, "03_04. 정렬", "=" * 3)
    print(data.sort_index())
    print(data.sort_index(ascending=False))

    print('1Key 정렬 : ',data.sort_values(by='증가', ascending=False))
    print('2Key 정렬 : ',data.sort_values(by=['사례수', '증가'], ascending=False))




    print('==컬럼 하나만==\n', data['index'])
    print('==범위 선택==\n', data[:3])

# pandas_03()

def pandas_04():
    """
        04. 데이터선택
            04_01. 기본 선택 방법

            04_02. loc, iloc 
                loc     : loc는 뒤의 선택값을 포함하여 가져온다. (이하)
                iloc    : iloc는 뒤의 선택값을 제외하고 가져온다 (미만)

            04_03. Boolean Indexing
                기본 사용방법
                isin    : 내가 조건을 걸고자 하는 값이 내가 정의한 list에 있을 경우 색인한다.

    """
    print("\n", "=" * 5, "04. ", "=" * 5)
    import pandas as pd
    import os

    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/data.csv', delimiter=',', encoding='CP949')
    print("\n", "=" * 3, "04_01.", "=" * 3)

    print(data['항목'])
    print(data[:3])
    print(data.head(3))

    print("\n", "=" * 3, "04_02.", "=" * 3)
    print('== 행전체, 컬럼 선택 ==\n', data.loc[:, ['항목', '사례수']])
    # loc는 6번 인덱스를 포함하고 가져온다.
    print('== 행일부, 컬럼 선택 ==\n', data.loc[3:6, ['항목', '사례수']])
    # loc는 6번 인덱스, 감소를 포함하고 가져온다.
    print('== 행일부, 컬럼 범위 ==\n', data.loc[3:6, '항목' : '감소'])

    print('== 행전체, 컬럼 선택 ==\n', data.iloc[:, [1, 4]])
    # iloc는 4번 인덱스를 포함하지 않고 가져온다.
    print('== 행전체, 컬럼 범위 ==\n', data.iloc[:, 1:4])

    print("\n", "=" * 3, "04_03.", "=" * 3)
    print('기본사용법\n', data[data['증가'] > 50])
    print('loc 사용법\n', data.loc[data['증가'] > 50, '항목':'감소'])

    my_isin = ['뉴스', '드라마', '예능']
    print('isin 사용법\n', data.loc[data['항목'].isin(my_isin), '항목':'감소'])
    
# pandas_04()

def pandas_05():
    """
        05. Null 비어있는 값
            
            Pandas 에서는 NaN = Not a Number 으로 표현된다.  
            05_01. 데이터 확인하기

            05_02. NaN 다루기 
                isna, isnull    전체 목록을 가져온다.   
                notnull         NaN이 아닌값만 가져온다.

    """
    print("\n", "=" * 5, "05. ", "=" * 5)
    import pandas as pd
    import os

    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/data.csv', delimiter=',', encoding='CP949')

    print("\n", "=" * 3, "05_01.", "=" * 3)

    print(data)
    print(data.info())

    print("\n", "=" * 3, "05_02.", "=" * 3)
    print('data.isna() \n', data.isna())
    print('data.isnull() \n', data.isnull())

    print('notnull() \n', data['사례수'].notnull())
    print('indexing \n', data[data['사례수'].notnull()])
    print('indexing \n', data.loc[data['사례수'].isnull(), '항목' :'사례수'])

    print("\n", "=" * 3, "05_03.", "=" * 3)
    
# pandas_05()

def pandas_06():
    """
        06. 복사, 추가, 삭제
            06_01. 복사
                프로그래밍에서 객체는 복사가 아닌 메모리주소 참조가 되기 때문에 
                복사하는 방법이 필요하다.
            06_02. 추가
                row, column 추가 방법 

            06_03. 삭제
                row
                    dropna() : 널값 전부 제거
    """
    print("\n", "=" * 5, "06. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/data.csv', delimiter=',', encoding='CP949')

    print("\n", "=" * 3, "06_01.", "=" * 3)
    df_01 = data
    df_01['사례수'][1] = 0
    print('data -  기존', data['사례수'][1], id(data))
    print('df_01 - 대입', df_01['사례수'][1], id(df_01))

    df_02 = df_01.copy()
    df_02['사례수'][1] = 9

    print('data -  기존', data['사례수'][1], id(data))
    print('df_01 - 대입', df_01['사례수'][1], id(df_01))
    print('df_02 - copy', df_02['사례수'][1], id(df_02))

    print("\n", "=" * 3, "06_02.", "=" * 3)
    dic_data = { '항목' : '12__01', '사례수': 95000, '증가':940, '감소':500}

    # row 추가
    data = data.append(dic_data, ignore_index=True)
    data.loc['99'] = {'항목' : '유튜브_02', '사례수': 95000, '증가':940, '감소':500}
    data.loc['100'] = ['유튜브_02',  95000, 940,  500, 99]
    print('row 추가 \n', data)

    # column 추가
    data['칼럼추가'] = '>.<'
    print('칼럼추가\n', data)

    # 내부 값 변경
    data.loc[data['증가'] > 40, '칼럼추가'] = 'O_O'
    print('내부 값 변경\n', data)

    print("\n", "=" * 3, "06_03.", "=" * 3)

    # NaN 전부 제거
    print('NaN값 확인 \n', data.isnull().sum())
    data = data.dropna()
    print(data)

    dic_data_01 = { '항목' : '유튜브_01', '사례수': 95000, '증가':940, '감소':500}
    dic_data_02 = { '항목' : '유튜브_02', '사례수': 95000, '증가':940, '감소':500}

    data = data.append(dic_data_01, ignore_index=True)
    data = data.append(dic_data_01, ignore_index=True)
    data = data.append(dic_data_02, ignore_index=True)
    data = data.append(dic_data_02, ignore_index=True)
    data = data.append(dic_data_02, ignore_index=True)

    print(data)
    print('중복행 확인 :', data.duplicated().sum())
    data = data.drop_duplicates()
    print(data)

    print('인덱스 순서로 삭제')
    data_index = data.loc[3:6].index
    print(data_index)
    data = data.drop(data_index)
    print(data)

    print('조건으로 삭제')
    data_index = data[data['증가'] < 40].index
    print(data_index)
    data = data.drop(data_index)
    print(data)

    
# pandas_06()

def pandas_07():
    """
        07. 통계값 다루기 
            describe()
                count   : 총 개수
                mean    : 평균
                std     : 표준 편차 (평균에 얼마나 떨어져 있는지)
                min     : 최소값
                25%     : 기준선 (25%)
                50%     : 기준선 (50%)
                75%     : 기준선 (75%)
                max     : 최대값
                
            07_01. 
                최소값  : min()
                최대값  : max()
                합계    : sum()
                평균    : mean()

            07_02. 분산 (variance - var), 표준 편차(standard deviation - std)
                분산 (variance - var)               : 데이터 - 평균 ** 2을 모두 합한 값
                    분산값이 크다 - 데이터의 간격이 넓게 분포되어 있다. 
                    데이터가 많을시 자주 사용되지는 않는다.

                표준 편차(standard deviation - std) : 표준편차는 분산의 루트값
                    데이터가 평균으로부터 얼마나 퍼져있는지 정도를 나타내는 지표

            07_03. 
                갯수    : count() 
                중앙값  : median()
                최빈값  : mode()

    """
    print("\n", "=" * 5, "07. ", "=" * 5)
    import pandas as pd
    import numpy as np
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/data.csv', delimiter=',', encoding='CP949')
    print(data.describe())

    print("\n", "=" * 3, "07_01.", "=" * 3)
    print('최소 min()   :', data['증가'].min())
    print('최대 max()   :', data['증가'].max())
    print('합계 um()    :', data['증가'].sum())
    print('평균 mean()  :', data['증가'].mean())

    print("\n", "=" * 3, "07_02.", "=" * 3)
    data_01 = np.array([1,3,5,7,9])
    data_02 = np.array([3,4,5,6,7])

    print(data_01)
    print('mean() :', data_01.mean())
    print('var()  :', data_01.var())
    print('std()  :', data_01.std())

    print(data_02)
    print('mean() :', data_02.mean())
    print('var()  :', data_02.var())
    print('std()  :', data_02.std())
    

    print("\n", "=" * 3, "07_03.", "=" * 3)
    data_01 = np.array([1,3,5,7,9])

    dic_data_01 = { '항목' : '유튜브_01', '사례수': 95000, '증가':100, '감소':500}

    dic_data_lst = []
    for cnt in range(1,20):
        if cnt < 2 :
            dic_data_01['증가'] = 100
            
        elif cnt < 5 : 
            dic_data_01['증가'] = 150
        elif cnt < 9 : 
            dic_data_01['증가'] = 200
        elif cnt < 14 : 
            dic_data_01['증가'] = 250
        else:
            dic_data_01['증가'] = 300
        dic_data = dic_data_01.copy()
        dic_data_lst.append(dic_data)


    print(dic_data_lst)

    data = data.append(dic_data_lst)
 
    print(data)
    print('count() :',  data.count())
    print('median() :', data['증가'].median())
    print('mode() :',   data['증가'].mode())

    
# pandas_07()

def pandas_08():
    """
        08. 피벗 테이블
            피벗테이블은 엑셀의 피벗테이블과 동일하다.
            데이터 열 중에서 두 개의 열을 각각 행 인덱스, 열 인덱스로 사용하여 데이터를 조회하여 펼쳐놓은 것을 의미한다.
                왼쪽    : 행 인덱스
                상단    : 열 인덱스

            08_01. 피벗테이블 사용법
                pd.pivot_table(data, index='확진일', columns='지역', values= '연번', aggfunc=['count']))
                    index               : 행 위치에 들어갈 열
                    columns             : 열 위치에 들어갈 열
                    values              : 데이터로 사용할 열
                    aggfunc=['count']   : 데이터 집계함수 

            08_02.
            08_03. 

    """
    print("\n", "=" * 5, "08. ", "=" * 5)
    import pandas as pd
    import numpy as np
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')

    print(data.head())
    print(data.info())
    print(data.describe())

    print("\n", "=" * 3, "08_01.", "=" * 3)
    pivot_data = pd.pivot_table(data, index='확진일', columns='지역', values= '연번', aggfunc=['count'])
    print(pivot_data)

    print(pivot_data['count', '강남구'].notnull())

    gangnam = pivot_data.loc[pivot_data['count', '강남구'].notnull(), ('count', '강남구')]
    print(gangnam)

# pandas_08()

def pandas_09():
    """
        09. GroupBy
            - GroupBy는 데이터를 그룹으로 묶어 분석할 때 활용합니다.
            - 그룹별 통계 및 데이터의 성질을 확인할때 활용합니다.
            09_01.
                총 개수   : count()
                최소값    : min()
                최대값    : max()
                합계      : sum()
                평균      : mean()
                분산      : var()
                표준편차   : std()
            09_02. Multi-Index (복합 인덱스)
                행 인덱스를 복합적으로 구성하고 싶은 경우는 인덱스를 리스트로 만들어 줍니다.
                data.groupby(['확진일', '지역'])
            09_03.
                unstack()       : Multi-index 데이터 프레임을 피벗 테이블로 변환(DataFrame)
                reset_index()   : Multi-index로 구성된 데이터 프레임의 인덱스를 초기화한다.

    """
    print("\n", "=" * 5, "09. GroupBy. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')
    print(data.head())

    print("\n", "=" * 3, "09_01.", "=" * 3)
    print(data.groupby('확진일')['연번'].count())

    print("\n", "=" * 3, "09_02.", "=" * 3)
    statistics = data.groupby(['확진일', '지역']).count()
    print('일, 지역별 통계 \n', statistics['연번'])



    print("\n", "=" * 3, "09_03.", "=" * 3)

    statistics = data.groupby(['확진일', '지역']).count()
    # AttributeError: 'DataFrameGroupBy' object has no attribute 'unstack'
    statistics = statistics.unstack('지역')
    print('unstack \n', statistics)

    df_01 = data.groupby(['확진일', '지역']).count()
    print(df_01)
    df_01 = df_01.reset_index()
    print(df_01)
    
# pandas_09()

def pandas_10():
    """
        10. Nan 값 처리

            10_01. fillna()
                Nan 값에 대하여 채워주는 함수
                1. fillna(값, inplace=True)
                    inplace=True 설정시 데이터 프레임에 값이 반영됩니다.
                2. copy후 값 넣기

            10_02. dropna()
                Nan 값에 대하여 제거해주는 함수
                how='any'   : 하나라도 Nan일 경우 제거 (default)
                how='all'   : 전부 Nan일 경우 제거
                axis=0      : 조건에 해당할시 행 제거 (default)
                axis=1      : 조건에 해당할시 열 제거

            10_03.drop_duplicates()
                중복을 제거하는 함수
                시리즈에서 혹은 데이터 프레임 전체에서 드랍이 가능하다.
                keep='first'    : 중복된 값중 앞에 나온값을 유지한다 (default)
                keep='last'     : 중복된 값중 마지막에 나온값을 유지한다

    """
    print("\n", "=" * 5, "10. ", "=" * 5)
    import pandas as pd
    import numpy as np
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')
    print(df_00.info())
    print("\n", "=" * 3, "10_01.", "=" * 3)
    df_01 = df_00.copy()
    print(df_01['환자번호'].fillna(-1))
    print(df_01)

    print(df_01['환자번호'].fillna(-1, inplace=True))
    print(df_01)

    df_02 = df_00.copy()
    df_02['환자번호'] = df_02['환자번호'].fillna(-1)
    print('df_02 : ', df_02)

    print("\n", "=" * 3, "10_02.", "=" * 3)

    df_03 = df_00.copy()
    df_03 = df_03[:100]
    print(df_03.info())
    print(df_03)
    print(df_03.dropna(axis=0))
    # how = any (default)
    print(df_03.dropna(axis=0, how='all'))
    df_03.iloc[99] = np.nan
    print(df_03)
    print(df_03.dropna(axis=0, how='all'))
    print(df_03.dropna(axis=1, how='all'))


    print("\n", "=" * 3, "10_03.", "=" * 3)

    df_04 = df_00.copy()

    # 데이터 프레임 전체
    print(df_04.drop_duplicates('확진일'))
    print(df_04.drop_duplicates('확진일', keep='last'))

    # 시리즈별
    print(df_04['확진일'].drop_duplicates())
    print(df_04['확진일'].drop_duplicates(keep='last'))

# pandas_10()


def pandas_11():
    """
        11. 삭제(drop), 합치기(concat)
            행과 열을 제거하는 함수
            option :
                axis        : 0, 1 (row del (Default) , column del)
                inplace     : True, False (실행시 반영, 반영X (Default) )
            11_01. drop
                1. drop(axis = 0) or drop()
                    row 삭제
                2. drop(axis = 1)
                    column 삭제

            11_02. concat (합치기)
                1. pd.concat([df_1, df_2], sort=False)
                        sort=False  : 미사용시 칼럼이 꼬인다 (row 기준으로 합칠때 사용)    :
                        axis=0, 1   : row (default), column

                2. reset_index() : concat시 인덱스 꼬일때 인덱스 초기화
                        drop=True   : Index 칼럼이 따로 안생기도록 해준다.
    """
    print("\n", "=" * 5, "11. ", "=" * 5)
    import pandas as pd
    import numpy as np
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')
    print("\n", "=" * 3, "11_01.", "=" * 3)
    df_01 = df_00.copy()
    df_01 = df_01[:100]

    print(df_01.head())
    print('df_01.drop(np.arange(0, 10), axis=0) \n', df_01.drop(np.arange(0, 10), axis=0))
    print("df_01.drop('환자번호', axis=1) \n", df_01.drop('환자번호', axis=1))
    print("df_01.drop(['환자번호','환자정보'], axis=1) \n", df_01.drop(['환자번호', '환자정보'], axis=1))

    print("\n", "=" * 3, "11_02.", "=" * 3)
    df_02_01 = pd.read_csv(path +'/file_data/data.csv', delimiter=',', encoding='CP949')
    df_02_02 = df_00.copy()
    df_02_02 = df_02_02[:20]

    # 예제 데이터 출력
    print(df_02_01.head())
    print(df_02_02.head())

    # row 단위로 합치기 1
    print(pd.concat([df_02_01, df_02_01.copy()], sort=False).reset_index())

    # row 단위로 합치기 2
    print(pd.concat([df_02_01, df_02_01.copy()], sort=False).reset_index(drop=True))

    # 칼럼 단위로 합치기
    # 행이 안맞으면 Nan이 나온다.
    print(pd.concat([df_02_01, df_02_02], axis=1, sort=False))

    print("\n", "=" * 3, "11_03.", "=" * 3)

# pandas_11()


def pandas_12():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            12. merge (병합)
        Describe
            merge   : 특정 기준(index)로 합치기
        sub Contents
            01. 기본 사용법
                pd.merge(left, right, on='기준column', how='left')
                    left    : DataFrame1
                    right   : DataFrame2
                    on      : 병합의 기준이 되는 column
                    how     : left, right, inner, outer ( join이랑 같다 )
            02. how 사용법
                inner   : 교집합 (Default)
                left    : 왼쪽 전체표시 오른쪽은 기준 값에 맞는 칼럼만 표시
                right   : 오른쪽 전체표시 왼쪽은 기준 값에 맞는 칼럼만 표시
                outer   : 합집합
            03. on 병합의 기준이 되는 column이 다를경우
                left_on     : 왼쪽 기준컬럼
                right_on    : 오른쪽 기준컬럼

    """
    print("\n", "=" * 5, "12. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data.csv', delimiter=',', encoding='CP949')

    #데이터 만들기
    df_01 = df_00.copy()
    df_02 = df_00.copy()
    df_02 = df_02[3:6]
    dic_data_01 = { '항목' : '유튜브_01', '사례수': 95000, '증가':940, '감소':500}
    df_02 = df_02.append(dic_data_01, ignore_index=True)

    print(df_01)
    print(df_02)

    print("\n", "=" * 3, "12_01. 기본 사용법", "=" * 3)
    print(pd.merge(df_01, df_02, on='항목'))

    print("\n", "=" * 3, "12_02. how 사용법", "=" * 3)
    print(pd.merge(df_01, df_02, on='항목', how="inner"))
    print(pd.merge(df_01, df_02, on='항목', how="left"))
    print(pd.merge(df_01, df_02, on='항목', how="right"))
    print(pd.merge(df_01, df_02, on='항목', how="outer"))

    print("\n", "=" * 3, "12_03. on 병합의 기준이 되는 column이 다를경우", "=" * 3)
    df_02.columns = ['종류', '사례수', '증가', '감소', '변화 없음']
    print(pd.merge(df_01, df_02, left_on='항목', right_on='종류', how="right"))


# pandas_12()



def pandas_13():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            13. Series (열)
        Describe
            Series는 어떤 데이터 타입이든 보유할 수 있는 label링 된 1차원 배열이다.
            - 정수, 문자열, float, 객체, 시간, 기타 등등을 포함한다.
            - 정수, label을 기반으로하는 인덱싱을 지원한다.
            - 인덱스를 포함한 연산에 사용 할 수 있는 다양한 메서드(method)를 지원한다.
            - 축 레이블은 인덱스와 같이 묶여서 참조될 수 있다.

        sub Contents
            01. type 변환하기
                df['key'].astype(str)

            02. dateTime
                시간 관련 변환
                year        : 년도
                month       : 월
                day         : 일
                hour        : 시간
                minute      : 분
                second      : 초
                dayofweek   : 요일
                weekofyear  : 주차
                
            03. on 병합의 기준이 되는 column이 다를경우
                left_on     : 왼쪽 기준컬럼
                right_on    : 오른쪽 기준컬럼

    """
    print("\n", "=" * 5, "13. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')

    #데이터 만들기
    df_01 = df_00.copy()
    print(df_01)

    print("\n", "=" * 3, "13_01.", "=" * 3)
    df_01['연번'] = df_01['연번'].astype(str)
    print(df_01.info())

    print("\n", "=" * 3, "13_02.", "=" * 3)
    df_01['확진일'] = pd.to_datetime(df_01['확진일'])

    df_01['확진_년'] = df_01['확진일'].dt.year
    df_01['확진_월'] = df_01['확진일'].dt.month
    df_01['확진_일'] = df_01['확진일'].dt.day
    df_01['확진_시간'] = df_01['확진일'].dt.hour
    df_01['확진_분'] = df_01['확진일'].dt.minute
    df_01['확진_초'] = df_01['확진일'].dt.second
    df_01['확진_요일'] = df_01['확진일'].dt.dayofweek
    df_01['확진_주차'] = df_01['확진일'].dt.weekofyear

    print(df_01)
    
    # print("dt.dayofweek", df_01['확진일'].dt.dayofweek[:4])
    # print("dt.dayofweek", df_01['확진일'].dt.weekofyaer)
    # print(df_01['확진일'].dt.day)
    # print(df_01['확진일'].dt.hour)
    # print(df_01['확진일'].dt.minute)
    print("\n", "=" * 3, "13_03.", "=" * 3)


# pandas_13()






def pandas_14():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            14. apply
        Describe
            Series나 DataFraem에 좀 더 구체적인 로직을 적용하고 싶은 경우 활용
                - apply를 적용하기 위해서는 함수가 먼저 정의되어야 합니다.
                - apply는 정의한 로직 함수를 인자로 넘겨줍니다.
        sub Contents
            01. 함수로 적용하기
            02. lambda 함수로 적용하기
            03. map값으로 매핑하기


    """
    print("\n", "=" * 5, "14. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data.csv', delimiter=',', encoding='CP949')

    #데이터 만들기
    df_01 = df_00.copy()
    print(df_01)

    def upper_50(x):
        if x > 50:
            return 1
        else:
            return 0
    print("\n", "=" * 3, "14_01.", "=" * 3)
    print(df_01.info())
    df_01['apply'] = df_01['증가'].apply(upper_50)
    print(df_01)

    print("\n", "=" * 3, "14_02.", "=" * 3)
    f = lambda x: 1 if x >= 50 else 0
    df_01['apply2'] = df_01['증가'].apply(f)
    print(df_01)

    print("\n", "=" * 3, "14_03.", "=" * 3)
    my_map = {
        '뉴스': 3,
        '예능': 1,
        '드라마': 2
    }
    df_01['apply3'] = df_01['항목'].map(my_map)
    print(df_01)


# pandas_14()


def pandas_15():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            15. 산술연산 
        Describe
            산술연산
        sub Contents
            01. Column, 숫자 연산
            02. 통계연산
            03. 데이터 프레임간의 연산

    """
    print("\n", "=" * 5, "15. ", "=" * 5)
    import pandas as pd

    data = ({
            '수학': [60, 70, 80, 90, 100],
            '과학': [40, 80, 60, 80, 50],
            '영어': [90, 80, 40, 30, 80],
            '국어': [99, 75, 85, 95, 75],
    })
    df_make_01 = pd.DataFrame(data)


    print("\n", "=" * 3, "15_01.", "=" * 3)
    print(df_make_01.info())

    # 연산에서 한쪽이 NAN일 경우 연산결과는 NAN이 됨
    # 없을경우에도 NAN

    print(df_make_01['수학'] + df_make_01['과학'])
    print(df_make_01['수학'] * 10)
    print(df_make_01)

    print("\n", "=" * 3, "15_02.", "=" * 3)
    df_make_01['총점 평균'] = df_make_01.mean(axis=1)
    avg_object = pd.DataFrame(df_make_01.mean(axis=0)).transpose()
    avg_object.index = ['과목평균']
    df_make_01 = df_make_01.append(avg_object)
    print(df_make_01)

    print("\n", "=" * 3, "15_03.", "=" * 3)
    data2 = ({
            '수학': [60, 70, 80, 90, 100],
            '영어': [90, 80, 40, 30, 80],
            '국어': [99, 75, 85, 95, 75],
            '과학': [40, 80, 60, 80, 50],
    })
    df_make_02 = pd.DataFrame(data2)
    # 연산 불가능할 경우 에러발생
    print(df_make_01 + df_make_02)

    # 칼럼 순서 상관없이 칼럼의 이름에 따라 자동으로 매핑된다.
    print(df_make_01 * 100)
    print((df_make_01 * df_make_02) / 100)


# pandas_15()



def pandas_16():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            16. select_dtypes
        Describe
            데이터 타입별 선택하는 메소드
            산술 가능한 문자열만 뽑아낼 수 있음.
        sub Contents
            01. include, exclude
            02. 대입
    """
    print("\n", "=" * 5, "16. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')
    print(df_00.info())

    print("\n", "=" * 3, "16_01.", "=" * 3)
    print(df_00.select_dtypes(include=object))
    print(df_00.select_dtypes(exclude=object) * 10)


    print("\n", "=" * 3, "16_02.", "=" * 3)
    df_00_obj_cols = df_00.select_dtypes(include=object)
    df_00_num_cols = df_00.select_dtypes(exclude=object)

    print(df_00_obj_cols)
    print(df_00_num_cols)

    print("\n", "=" * 3, "16_03.", "=" * 3)

    return

# pandas_16()



def pandas_17():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            17. 원핫인코딩 (One-hot-encoding)
        Describe
            한개의 요소는 True, 나머지 요소는 False로 만들어 주는 기법
        sub Contents
            01. map 만들기
    """
    print("\n", "=" * 5, "17. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')


    print("\n", "=" * 3, "17_01.", "=" * 3)
    df_00_obj_cols = df_00.select_dtypes(include=object)
    df_03_area_sum = df_00_obj_cols.groupby(['여행력']).count()
    df_index_list = df_03_area_sum.index.to_list()
    df_map = {string: i for i, string in enumerate(df_index_list)}
    print(df_map)

    df_00['여행지역'] = df_00['여행력'].map(df_map)

    print('indexing \n', df_00.loc[df_00['여행지역'].notnull(), ['연번', '확진일', '지역', '여행력']])
    print(df_00['여행지역'].value_counts())

    print("\n", "=" * 3, "17_02.", "=" * 3)
    data = pd.get_dummies(df_00['여행지역'], prefix='여행지')
    # 데이터별로 1 혹은 0으로 전체 표시 가능  많으면 안보이지만..
    df_00['여행지역'].value_counts().to_excel('inventors.xlsx')

    print(data)
    print("\n", "=" * 3, "17_03.", "=" * 3)

# pandas_17()


def pandas_fn_rename():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            칼럼 바꾸는 메소드
        Describe
            칼럼 바꾸는 메소드
        sub Contents
            01. df.rename(columns={'before Column":'after Column'});
    """
    print("\n", "=" * 5, "fn_rename. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')

    print("\n", "=" * 3, "fn_rename_01.", "=" * 3)
    print(df_00.info())
    df_00 = df_00.rename(columns={'여행력':'여행지'})
    print(df_00.info())

# pandas_fn_rename()

def df_pandas_ex_01():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            연습문제 1번
        Describe
            코로나 데이터로 이것저것 해보기
        sub Contents
            01.
    """
    # 속한 지역구에 대한 전체 합산
    import pandas as pd
    import numpy as np
    import os

    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')
    df_03 = df_00.copy()
    df_03 = df_03.loc[:, '연번':'지역']
    df_03_day_sum = df_03.groupby(['확진일'])['연번'].count()
    df_03_area_sum = df_03.groupby(['지역'])['연번'].count()
    df_03_day_area_sum = df_03.groupby(['확진일', '지역'])['연번'].count()

    print(df_03_day_sum)
    print(df_03_area_sum)
    print(df_03_day_area_sum)
    print(type(df_03_area_sum))

    new_col = ['연번', '확진일', '지역합산', '일 합산', '비율', '지역']
    df_03.columns = new_col
    df_03['지역합산'] = df_03['지역'].apply(lambda x: df_03_area_sum[(x)])
    df_03['일 합산'] = df_03['확진일'].apply(lambda x: df_03_day_sum[(x)])
    df_03['일별 지역합산'] = df_03.apply(lambda x: df_03_day_area_sum[(x['확진일'], x['지역'])], axis=1)

    df_03['비율'].fillna(df_03['일별 지역합산'] / df_03['지역합산'], inplace=True)

    print(df_03)

# df_pandas_ex_01()


def pandas_ex02():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            연습문제 2번
        Describe
            부동산 데이터로 이것저것 해보기
        sub Contents
            01.
    """
    print("\n", "=" * 5, "ex02. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/주택도시보증공사_전국 신규 민간아파트 분양가격 동향_20200430.csv', delimiter=',', encoding='CP949')

    print(df_00.info())
    df_00 = df_00.rename(columns={'분양가격(㎡)': '분양가격', '규모구분': '규모', '지역명':'지역'})
    print(df_00.info())

    print("\n", "=" * 3, "ex02_01.", "=" * 3)
    print(df_00.describe())
    """
        File "pandas\_libs\lib.pyx", line 615, in pandas._libs.lib.astype_intsafe
        ValueError: invalid literal for int() with base 10: '  '
        df_00['분양가격'].astype(int)
        그냥 사용시 공백 에러 발생함.
    """
    print(df_00.describe())
    print(df_00.loc[df_00['분양가격'] == '  '])

    # 문자열로 변환해서 공백 제거
    df_00['분양가격'] = df_00['분양가격'].str.strip()
    df_00.loc[df_00['분양가격'] == '', '분양가격'] = 0

    # NAN 0으로 변경
    df_00['분양가격'] = df_00['분양가격'].fillna(0)
    print(df_00.loc[df_00['분양가격'] == '  '])

    # 타입 변경
    df_00['분양가격'] = df_00['분양가격'].astype(int)
    # 이건 그냥 변경
    df_00['규모'] = df_00['규모'].str.replace('전용면적 ', '')
    print(df_00.describe())
    print(df_00)

    print("\n", "=" * 3, "ex02_02. 평균계산", "=" * 3)
    print(df_00.groupby('지역')['분양가격'].mean())

    #불필요한 데이터 제거
    df_00_remove_idx = df_00.loc[df_00['분양가격'] < 100].index
    print('삭제전 \n', df_00.count())
    df_00 = df_00.drop(df_00_remove_idx, axis=0)
    print('삭제후 \n', df_00.count())

    # 지역별 평균, 데이터 개수, 최대값
    print(df_00.groupby('지역')['분양가격'].mean())
    print(df_00.groupby('지역')['분양가격'].count())
    print(df_00.groupby('지역')['분양가격'].max())

    # 연도별 평균, 데이터 개수, 최대값
    print(df_00.groupby('연도')['분양가격'].mean())
    print(df_00.groupby('연도')['분양가격'].count())
    print(df_00.groupby('연도')['분양가격'].max())

    # 지역, 연도 평균
    print(df_00.groupby(['지역', '연도'])['분양가격'].mean())

    # 연도, 규모별 평균
    print(df_00.groupby(['연도', '규모'])['분양가격'].mean())

    print("\n", "=" * 3, "ex02_03. 피벗 테이블", "=" * 3)

    print(pd.pivot_table(df_00, index='연도', columns='규모', values='분양가격'))
    print(pd.pivot_table(df_00, index='연도', columns='규모', values='분양가격'))


# pandas_ex02()

def pandas_temp():
    """
        subject
            Data analysis module
        topic
            Pandas 모듈
        content
            ex. select_dtypes
        Describe
            데이터 타입별 선택하는 메소드
        sub Contents
            01.



    """
    print("\n", "=" * 5, "temp. ", "=" * 5)
    import pandas as pd
    import os
    # 절대경로 얻기
    path = os.path.dirname(os.path.abspath(__file__))
    df_00 = pd.read_csv(path +'/file_data/data2.csv', delimiter=',', encoding='CP949')

    print("\n", "=" * 3, "temp_01.", "=" * 3)
    print("\n", "=" * 3, "temp_02.", "=" * 3)
    print("\n", "=" * 3, "temp_03.", "=" * 3)