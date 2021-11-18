"""
    02_data_structure
        01. list
        02. tuple
        03. Dictionary
        ▶ 04. Set
"""

"""
      04. Set
            집합에는 순서가 없다.
            04_01. Set 집합 타입 기본
            04_02. 메소드
            04_03. 사용 예시
            help(dict)
            
      print(help(tuple)) DOC 호출
"""


def data_structure_04():
    print('='*20, '04. Set', '='*20)

    lst = []
    lst.append('04_01. Set 집합 타입 기본')
    lst.append('04_02. 메소드')
    lst.append('04_03. 사용 예시')

    for i in range(1, len(lst)+1):
        print(i, ' : ', lst[i-1])
    lst_cnt = int(input('숫자를 입력하세요 : '))

    print('='*5, lst[lst_cnt-1], '='*5)

    if lst_cnt == 1:
        print(help(set))
        s = {1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 8, 8, 8, 8}
        print(s)

        a = {1, 2, 3, 4}
        b = {4, 5, 6, 7}
        print('합칩합 : ', a | b)
        print('교집합 : ', a & b)
        print('차집합 a-b : ', a - b)
        print('차집합 b-a : ', b - a)
    elif lst_cnt == 2:
        print(1)
        s =  {1, 2, 3, 4, 5}
        s.add(6)
        print('s =  {1, 2, 3, 4, 5} :', type(s))
        d = {}
        print('d = {} : ', type(d))

    elif lst_cnt == 3:
        my_friends = {'A', 'B', 'C'}
        your_friends = {'B', 'C', 'D', 'E'}

        print('my_friends & your_friends : ', my_friends & your_friends)

        f = ['apple', 'banana', 'apple']
        kind = set(f)
        print("['apple', 'banana', 'apple'] => :", kind)

    elif lst_cnt == 4:
        print(1)
    else:
        print('선택항목은 없는 항목')


data_structure_04()




