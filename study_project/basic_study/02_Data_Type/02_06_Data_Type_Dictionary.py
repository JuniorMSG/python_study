"""
Author  : SG
DATE    : 2021-05-22
02_Data_Type

02_01. Integer      (숫자형)
02_02. String       (문자열)
02_03. Boolean      (참&거짓 자료형)
02_04. List         (리스트)
02_05. Tuple     (수정 불가능 리스트)
★ 02_06. Dictionary   (HashMap - Key- Map Type)
02_07. Set          (집합 자료형)
02_08. Variable     (변수)

★ 02_06. Dictionary   (HashMap - Key- Map Type)
    01. Dictionary 자료형 이란?
    02. Dictionary 자료형 선언방법
    03. Dictionary 쟈료형 사용방법
    04. 내장함수(built-in Function)
"""

"""
01. Dictionary 자료형 이란?
    Dictionary 자료형은 Key = Value 구조로 이루어진 자료형이다. 
    다른 자료형과 다르게 순차적으로 이뤄지지 않아서 Key값으로 데이터를 찾는 특징을 가지고 있다.
    
    EX) "이름" : "SS" , "주민등록번호" : "999999-11111111" 형태로 만들어짐
"""


"""
02. Dictionary 자료형 선언방법
    {"key":"value"} 형태로 선언하면 된다.
"""
print("\n", "=" * 5, "02. Dictionary 자료형 선언방법", "=" * 5)

dic1 = {"name" : "python", "version" : "3.6.12", "Language Type" : "Script"}
print('dic1["name"]             : ', dic1["name"])
print('dic1["version"]          : ', dic1["version"])
print('dic1["Language Type"]    : ', dic1["Language Type"])

data_lst = [1,2,3]
data_tuple  = (4,5,6)
data_num = 999

dic2 = { 
    "data" : dic1, 
    "data_lst":data_lst, 
    "data_tuple":data_tuple, 
    "data_num" : data_num
    }

print('dic2["data"]         :', dic2["data"])
print('dic2["data_lst"]     :', dic2["data_lst"])
print('dic2["data_tuple"]   :', dic2["data_tuple"])
print('dic2["data_num"]     :', dic2["data_num"])

"""
03. Dictionary 쟈료형 사용방법
    1. 추가하기
    2. 삭제하기
    3. 사용하기
    4. 주의사항
        - 키가 중복됐을 경우 1개를 제외한 나머지 값이 무시된다.
        - 값이 변할 수 있는 요소는 Key로 사용할 수 없다. List - unhashable type: 'list' 에러가 발생한다.
"""
print("\n", "=" * 5, "03. Dictionary 쟈료형 사용방법", "=" * 5)

dic_03 = { 
    "data" : dic1, 
    "data_lst":data_lst, 
    "data_tuple":data_tuple, 
    "data_num" : data_num,
    4 : "NUMBER4"
    }

print("\n", "=" * 3, "1. 추가하기", "=" * 3)
dic_03[5] = "NUMBER5"
print("dic_03[5]        :", dic_03[5])
dic_03["add_lst"] = [9,5,4]
print("dic_03[add_lst]  :", dic_03["add_lst"])

print("\n", "=" * 3, "2. 삭제하기", "=" * 3)
print("dic_03            :", dic_03)

del dic_03["add_lst"]
print('dic_03["add_lst"] :', dic_03)

del dic_03["data"]
print('dic_03["data"]    :', dic_03)


print("\n", "=" * 3, "3. 사용하기", "=" * 3)
print('dic_03["data_lst"]   :', dic_03["data_lst"]  )
print('dic_03["data_tuple"] :', dic_03["data_tuple"])
print('dic_03["data_num"]   :', dic_03["data_num"]  )
print('dic_03[4]            :', dic_03[4]           )
print('dic_03[5]            :', dic_03[5]           )


print("\n", "=" * 3, "4. 주의사항", "=" * 3)

dic_04 = {(1,2,3):6 , (4,4,4):12}
# dic_05 = {[1,2,3]:5} ERROR CODE
print(dic_04[(1,2,3)])
print(dic_04[(4,4,4)])

# print(dic_05[[1,2,3]]) ERROR CODE

"""
04 내장함수(built-in Function)
    1. 요소를 얻는 방법 
        Keys    : Key 전체 리턴
        values  : value 전체 리턴
        items   : key, value 쌍 리턴
        get     : key로 value값 얻기
    2. 요소를 조사하는 방법 
        in      : 해당 Key가 딕셔너리 안에 있는지 조사하기
"""

print("\n", "=" * 5, "04. 내장함수(built-in Function)", "=" * 5)

dic_04 = { 
    "data" : dic1, 
    "data_lst":data_lst, 
    "data_tuple":data_tuple, 
    "data_num" : data_num,
    4 : "NUMBER4"
    }
    
print("\n", "=" * 3, "1. 요소를 얻는 방법", "=" * 3)
print('dic_04.keys()            :', dic_04.keys())
print('dic_04.values()          :', dic_04.values())
print('dic_04.items()           :', dic_04.items())
print('dic_04.get("data_tuple") :', dic_04.get("data_tuple"))
print('dic_04.["data_tuple"]    :', dic_04["data_tuple"])

print("\n", "=" * 3, "2. 요소를 조사하는 방법", "=" * 3)


print('"data_num" in dic_04  :', "data_num" in dic_04)
print('"data_num1" in dic_04 :', "data_num1" in dic_04)
print('4 in dic_04           :', 4 in dic_04)
print('5 in dic_04           :', 5 in dic_04)