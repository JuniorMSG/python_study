"""
    04_fn
        함수관련된 것 전부
        04_01. 함수 기본

"""

def print_set(data):
    print('='*5, data, '='*5)

def study_lst():
    print('='*20, '04_fn', '='*20)

    lst = []
    lst.append('04_01 함수 기본 ')

    for i in range(1, len(lst)+1):
        print(i, ' : ', lst[i-1])
    lst_cnt = int(input('숫자를 입력하세요 : '))

    print('='*5, lst[lst_cnt-1], '='*5)

    if lst_cnt == 1:
        def say_somtthing():
            print('hi')
        print(type(say_somtthing))
        def color_print(color):
            if color == 'red':
                return 'red_color'
            else:
                return 'no_color'

        result = color_print('red')
        print(result)
        result = color_print('blue')
        print(result)

        data = ('함수의 인수와 반환값의 선언, '
                '문자열이 들어가도 에러는 안나고 표시용임..')
        print_set(data)

        def add_nul(a: int, b: int) -> int:
            return a+b
        r = add_nul(5, 10)
        print(r)
        r = add_nul('a', 'b')
        print(r)

        data = '키워드를 지정해서 사용하면 순서와 상관없이 실행 할 수 있음.'
        print_set(data)

        def menu(entree='beef', drink='wine', dessert='ice'):
            print(entree)
            print(drink)
            print(dessert)
        menu(entree='chicken', dessert='ice', drink='beer')

        # 리스트형태는 디폴트 인수로 사용하면 안된다.
        print_set('리스트형태는 디폴트 인수로 사용하면 안된다.')
        def test_func(x, lst=[]):

            lst.append(x)
            return lst

        y = [1, 2, 3]
        r = test_func(100, y)
        print(r)

        y = [1, 2, 3]
        r = test_func(200, y)
        print(r)

        r = test_func(100)
        print(r)
        r = test_func(100)
        print(r)

        def test_func(x, lst=None):
            # 리스트형태는 디폴트 인수로 사용하면 안된다.
            if lst is None:
                lst = []
            lst.append(x)
            return lst

        r = test_func(100)
        print(r)
        r = test_func(100)
        print(r)

        # 위치 인수의 튜플화
        print_set('위치 인수의 튜플화 *args에서 튜플형태로 묶어서 출력해주는것.')

        def print_lst(word, *args):
            print('word =', word)
            for arg in args:
                print(arg)

        print_lst(1, 2, 3, 4, 5, 6, 7)

        # 키워드 인수의 사전화화
        def menu(food, *args, **kwargs):
            print('food :', food)
            print('args :', args)
            print('kwargs :', kwargs)
            for k, v in kwargs.items():
                print(k, v)
        d = {
            'entree' : 'beef',
            'drink' : 'ice coffee',
            'dessert' : 'ice'
        }
        menu('banana', 'apple', 'orange', entree='beef', drink='coffee')

        # DOcstrings 란?
        def example_func(param1, param2):
            """
                example_func with types documented in the docstring
            :arg
                :param param1 (int): The first parameter.
                :param param2 (str): The second parameter
            :return
                bool: The return value. True for success, False otherwise.
            """
            print(1234)
        print('help(example_func) : ', help(example_func))

        # inner function 내부 함수

        def outer(a, b):

            def plus(c, d):
                return c+d
            r = plus(a, b)
            print(r)

        outer(1, 24)

    elif lst_cnt == 2:
        print(1)
    elif lst_cnt == 3:
        print(1)
    elif lst_cnt == 4:
        print(1)
    else:
        print('선택항목은 없는 항목')

study_lst()



