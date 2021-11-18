"""
    02_data_structure
        01. list
        02. tuple
        ▶ 03. Dictionary
        04. Set
"""

"""
      03. Dictionary
            Key로 검색해서 값을 가져오는 경우 ( Key - value 구조) Dic 타입이 편한다
            HashTable을 가지고 있어서 키를 알면 금방 값을 알 수 있다.
            
            03_00. 사전형 Help
            03_01. 사전형 기본
            03_02. 사전형 메소드
            03_03. 사용 예시
            help(dict)
            
      print(help(tuple)) DOC 호출
"""


def data_structure_03():
    print('='*20, '03. Dictionary', '='*20)

    lst = []
    lst.append('03_01. 사전형 Help')
    lst.append('03_02. 사전형 기본')
    lst.append('03_03. 사전형 메소드')
    lst.append('03_04. 사용 예시')

    for i in range(1, len(lst)+1):
        print(i, ' : ', lst[i-1])
    lst_cnt = int(input('숫자를 입력하세요 : '))

    print('='*5, lst[lst_cnt-1], '='*5)

    if lst_cnt == 1:
        print(lst[lst_cnt])
        print(help(dict))
    elif lst_cnt == 2:
        print(lst[lst_cnt])

        d = {'x': 10, 'y': 20}
        print(type(d))
        print('d["x"] : ', d['x'])
        print('d["y"] : ', d['y'])

        d['x'] = 100
        print(d)

        d['z'] = 200
        print(d)

        d['1'] = 10000
        print(d)

        data = dict(a=10, b=20)
        print(data)
        data = dict([('a', 30), ('b', 40)])
        print(data)

        print('대입시 생기는 문제')
        data_x = data
        data_x['a'] = 999
        print(data, id(data))
        print(data_x, id(data_x))

        print('리스트와 마찬가지로 copy를 해서 사용해야함.')
        data_y = data.copy()
        data_y['a'] = 9999
        print(id(data), data, id(data_y), data_y)


    elif lst_cnt == 3:
        print(lst[lst_cnt])
        d = {'x': 100, 'y': 50, 'z': 10}
        print(d)
        print(d.keys())
        print(d.values())

        d2 = {'x': 1000, 'y': 500}
        print(d2)

        print("'x' in d2", 'x' in d2)
        print("'y' in d2", 'y' in d2)
        print("'z' in d2", 'z' in d2)

        d.update(d2)
        print(d)

        print(d.get('x'))
        # print(d.get['z'])

        print("d.pop('x') : ", d.pop('x', d), d)

        del d['z']
        print("del d['z'] : ", d)

        d.clear()
        print("d.clear() : ", d)

    elif lst_cnt == 4:
        fruits = {
            'apple': 100,
            'banana': 200,
            'orange': 300
        }
        print('apple', fruits['apple'])
    else:
        print('선택항목은 없는 항목')


data_structure_03()




