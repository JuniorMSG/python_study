"""
    Study Basic Python grammar
    DataType : String Type

    author : MSG
"""

def list_type():
    a = [1,2,3]
    b = [4,5,6,'String', 0.5, [99,88,77, 'String2']]
    c = a + b
    print("a     : ", a)
    print("b     : ", b)
    print("a + b : ", a+b)
    print("a * 2 : ", a*2)

    print('===== 인덱싱 =====')
    c = a + b
    print("default  :   c       : ", c)
    print("Indexing :   c[0]    : ", c[0])
    print("Indexing :   c[4]    : ", c[4])
    print("Indexing :   c[8][2] : ", c[8][2])
    print("Indexing :   c[8][3][2] : ", c[8][3][2])
    print("Slicing  :   c[0:4]  : ", c[0:4])
    print("Slicing  :   c[4:]   : ", c[4:])

    print('===== 삭제 =====')
    del(c[0])
    print("del(c[0]) : ", c)
    del(c[5])
    print("del(c[5]) : ", c)

    return


def list_type_built_in_fn():
    result = [3,2,1]

    print("result : ", result)
    # append(var)   :    요소를 뒤에 추가
    result.append(4)
    print("result.append(4) : ", result)

    # sort()    :    요소를 정렬
    result.sort()
    print("result.sort()    : ", result)

    # reverse() :  뒤집음
    result.reverse()
    print("result.reverse() : ", result)

    # index(var) :  입력값의 인덱스를 반환
    print("result.index(2)  : ", result.index(2))

    # insert(index, var) :  특정 위치에 요소를 추가
    result.insert(3,9)
    print("result.insert(3,9) : ", result)

    # remove(var) :  입력값을 삭제 (첫번째로 찾은 값)
    result.remove(9)

    # pop : 마지막 요소를 꺼내고 삭제
    result = [1,2,3,'String', 'int', 'list', 'tuple', 1,2,3,1,2,3,1,2,3]
    print("result       : ", result); result.pop()
    print("result.pop() : ", result); result.pop()
    print("result.pop() : ", result); result.pop()
    print("result.pop() : ", result); result.pop()

    # count : 입력값의 갯수 반환
    print("result           : ", result)
    print("result.count(1)  : ", result.count(1))


def tuple_type():
    result = (1, 2, 3, 4, 5, 1, 2, 3, 4, 5)

    # (X) append(var)   :    요소를 뒤에 추가
    # (X) sort()    :    요소를 정렬
    # (X) reverse() :  뒤집음
    # (X) index(var) :  입력값의 인덱스를 반환
    # (X) insert(index, var) :  특정 위치에 요소를 추가
    # (X) remove(var) :  입력값을 삭제 (첫번째로 찾은 값)
    # (X) data.insert(1,2)

    # (O) count : 입력값의 갯수 반환
    print("result           : ", result)
    print("result.count(1)  : ", result.count(1))

def dictionary_type():
    print("=" * 15, "dictionary_type", "=" * 15)
    data = {"name":"ms", "age":18, "list":[1,2,3], "Tuple":(1,2,3,4)}

    print(data)
    print("Indexing :   data['name']    : ", data['name'])
    print("Indexing :   data['age']     : ", data['age'])
    print("Indexing :   data['list']    : ", data['list'])



def dictionary_built_in_fn():
    print("=" * 15, "dictionary_built_in_fn", "=" * 15)

    data = {"name": "ms", "age": 18, "list": [1, 2, 3], "Tuple": (1, 2, 3, 4)}

    # keys()    : 딕셔너리의 key를 dict_keys 객체로 반환
    print("data.keys() : ", data.keys())

    # values()  : 딕셔너리의 value를 dict_values 객체로 반환
    print("data.keys() : ", data.values())

    # items()  : 딕셔너리의 key, value  dict_items 객체로 반환
    print("data.keys() : ", data.items())

    # get()  : key에 대한 값을 반환 (값이 없는경우 에러가 나는데 기본값 지정 가능)
    print("data.get('ttt', 111) : ", data.get('ttt'))
    print("data.get('ttt', 111) : ", data.get('ttt', 111))

    # del()  : key삭제가능
    del(data['name'])
    print("del(data['name']) : ", data)

    # 추가
    data['name'] = 'ms'
    print("data['name'] = 'ms' : ", data)

    return

# list_type()
# list_type_built_in_fn()
# tuple_type()
# dictionary_type()
dictionary_built_in_fn()

