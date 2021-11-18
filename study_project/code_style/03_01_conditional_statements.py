"""
    03_conditional_statements
        03_01. if문 기본

"""

import def_set as ds

def default_01():
    print('='*20, '03. 03_conditional_statements', '='*20)

    lst = []
    lst.append('03_01. if문 기본')
    lst.append('03_02. 논리 연산자')
    lst.append('03_03. in, not, True, False, None')
    lst.append('03_04. 반복문 while')
    lst.append('03_05. 반복문 for')
    lst.append('03_06. 같이 잘 사용되는 함수들')

    for i in range(1, len(lst)+1):
        print(i, ' : ', lst[i-1])
    lst_cnt = int(input('숫자를 입력하세요 : '))

    print('='*5, lst[lst_cnt-1], '='*5)

    if lst_cnt == 1:
        """
            파이썬에선 단계 구분을 indent로 하니까 
            잘 맞춰서 작성해야함.
        """
        x = 10
        if x < 0:
            print(True)
        elif x == 0:
            print('zero')
        elif x == 10:
            print('10')


        print(1)
    elif lst_cnt == 2:
        a, b = 1, 1
        print(a == b)
        print(a != b)
        print(a < b)
        print(a > b)
        print(a <= b)
        print(a >= b)
        print(a > 0 and b > 0)
        print(a > 0 or b > 0)

    elif lst_cnt == 3:
        y = [1, 2, 3]
        x = 1
        print(x in y)
        print(100 not in y)

        # 숫자의 경우는 !=을 사용하는게 맞음.
        a, b = 1, 2
        print('not a == b =>', not a == b)
        print('a != b =>', a != b)

        # True 판정
        # True  : 무언가 있으면 True취급됨.
        # False : False, 0, 0.0, '', [], (), {}, set()등 빈형태는 False 취급됨

        print("False, 0, 0.0, '', [], (), {}, set()등 빈형태는 False 취급됨")
        is_ok = 100
        if is_ok:
            print('OK!')
        else:
            print('NO')

        is_empty = None
        print(is_empty)

        if is_empty is not None:
            print('None!!')

        # is Object간 비교임 == 값간 비교임
        print('1 == True :', 1 == True)
        print('1 is True :', 1 is True)
        print('True is True :', True is True)
        print('None is None :', None is None)

    elif lst_cnt == 4:
        count = 0
        while count < 5:
            print(count)
            count += 1

        print('='*20)
        count = 0
        while True:
            print(count)
            if count >= 5:
                # break 종료
                break
            if count == 2:
                count += 2
                # while 처음부터 다시 실행
                continue
            count += 1
        else:
            # 다른 프로그램에서 finally랑 비슷함 break로 끝날경우엔 실행 안됨.
            print('done')

    elif lst_cnt == 5:

        some_list = [1, 2, 3, 4, 5]

        # iterate라고 함.
        for i in some_list:
            print(i)

        for s in 'abcde':
            print(s)

        for word in ['My', 'name', 'sg']:
            if word == 'name':
                continue
            print(word)
        else:
            print('else')

    elif lst_cnt == 6:

        num = [1, 2, 3, 4, 5]

        # range
        # _ 언더스코어를 사용하면 이 인덱스는 안쓴다는 명시적 표현임
        for _ in range(2, 10, 3):
            print('Hello')

        # enumerate 함수
        # 인덱스와 같이 사용하고 싶을때.
        ds.print_set('enumerate 함수 : 인덱스와 같이 사용하고 싶을때.')
        for i, fruit in enumerate(['apple', 'banana', 'orange']):
            print(i, fruit)

        # zip 함수
        ds.print_set('zip 함수 여러개의 리스트에서 뽑고 싶을때')
        days = ['Mon', 'Tue', 'Wed']
        fruits = ['apple', 'banana', 'orange']
        drinks = ['coffee', 'tea', 'beer']

        for day, fruit, drink in zip(days, fruits, drinks):
            print(day, fruit, drink)

        #  dic_type process for
        ds.print_set('dic_type process for')
        d = {'x': 100, 'y': 200}
        for k, v in d.items():
            print(k, ':', v)
        print(d.items())



    else:
        print('선택항목은 없는 항목')


default_01()




