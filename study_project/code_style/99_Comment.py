"""
      99. Comment


"""


def comment_01():
    print('='*20, '99. Comment', '='*20)
    """
        test
        test
        test
        test
    """

    lst = []
    lst.append('99_01. 암묵적인 룰')
    lst.append('99_02. 기타 팁 - 한줄이 길어질 경우')

    for i in range(1, len(lst)+1):
        print(i, ' : ', lst[i-1])
    lst_cnt = int(input('숫자를 입력하세요 : '))

    print('='*5, lst[lst_cnt-1], '='*5)

    if lst_cnt == 1:
        # 변수의 커맨트는 변수 위에 사용한다.
        a = '변수의 커맨트는 변수 위에 사용한다'
        print(a)
    elif lst_cnt == 2:
        
        print('한줄에 80자 이상은 에러가 발생함.')
        
        s = 'aaaaaaaaaaaaaa' \
            + 'bbbbbbbbbbbbbbb'
        print(s)
        x = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 \
            + 1 + 1 + 1 + 1 + 1 + 1 + 1
        print(x)

        s = ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' +
            'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        print(s)

        x = (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1
            + 1 + 1 + 1 + 1 + 1 + 1 + 1)
        print(x)

    elif lst_cnt == 3:
        print(1)
    elif lst_cnt == 4:
        print(1)
    else:
        print('선택항목은 없는 항목')


comment_01()




