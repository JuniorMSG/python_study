class testClass:
    def sum(self, a, b):
        return a+b

    def sub(self, a, b):
        return a-b
    pass


class bulitInFn:
    # abs : 절대값 리턴
    print("=" * 30)
    print("abs : 절대값 리턴 " , len("abs : 절대값 리턴"))
    print("=" * 30)
    print(abs(-3), abs(3))

    # all : 반복 가능한 자료형을 입력 인수로 받음 참이면 True, 거짓이 있으면 False
    print("=" * 30)
    print("all : 반복 가능한 자료형을 입력 인수로 받음 참이면 True, 거짓이 있으면 False")
    print("=" * 30)
    print("all([1,2,3])     :   ", all([1,2,3]))
    print("all([1,2,3,0])   :   ", all([1,2,3,0]))

    # any : 값중 하나라도 참이 있을경우 True
    print("=" * 30)
    print("any : 값중 하나라도 참이 있을경우 True")
    print("=" * 30)
    print("any([1,2,3,4])   :   ", any([1,2,3,4]))
    print("any([0, ''])     :   ", any([0,'']))

    # chr : 아스키 코드값을 입력받아 코드에 해당하는 문자 출력 ord : 반대
    print("=" * 30)
    print("chr :아스키 코드값을 입력받아 코드에 해당하는 문자 출력,  ord : 반대")
    print("=" * 30)
    print("chr(92)  :   ", chr(92))
    print("chr(43)  :   ", chr(43))

    print("ord(chr(92)) : ", ord(chr(92)))
    print("ord(chr(43)) : ", ord(chr(43)))

    # dir : 객체가 자체적으로 가지고 있는 변수나 함수를 보여줌
    print("=" * 30)
    print("dir : 객체가 자체적으로 가지고 있는 변수나 함수를 보여줌")
    print("=" * 30)

    print("dir([1,2,3])     : ", dir([1, 2, 3]))
    print("dir({1,2})       : ", dir({1, 2}))
    print("dir({'1':'2'})   : ", dir({'1': '2'}))

    # divmod : a를 b로 나눈 몫과 나머지를 튜블형태로 리턴
    print("=" * 30)
    print("divmod : a를 b로 나눈 몫과 나머지를 튜블형태로 리턴")
    print("=" * 30)

    print("divmod(7,3)      :   ", divmod(7,3))
    print("divmod(1.5,0.3)  :   ", divmod(1.5,0.3))

    # enumerate : 순서가 있는 자료형을 인덱스 값을 포함하는 enumerate 객체로 리턴
    print("=" * 30)
    print("enumerate : 순서가 있는 자료형을 인덱스 값을 포함하는 enumerate 객체로 리턴")
    print("반복되는 구간에서 객체의 인덱스값이 필요할때 사용")
    print("=" * 30)

    item = [1,2,3,4,5]
    en_item = enumerate(item)

    for i, data in en_item:
        print("인덱스 :", i ,"값 :", data)

    # eval : 문자열을 입력으로 받아 문자열을 실행한 결과값을 리턴
    print("=" * 30)
    print("eval : 문자열을 입력으로 받아 문자열을 실행한 결과값을 리턴")
    print("=" * 30)

    print("eval('1+2') : ", eval('1+2'))
    print("eval('a'+'b') : ", eval("'a'+'b'"))
    print("eval('divmode(4,3)') : ", eval('divmod(4,3)'))


    # filter(함수이름, 반복가능한 데이터)
    # 데이터가 함수에 들어가서 참일경우만 돌려줌
    print("=" * 30)
    print("filter(함수이름, 반복가능한 데이터) : 데이터가 함수에 들어가서 참일경우만 돌려줌")
    print("=" * 30)
    def true_return(data):
        return data > 0

    print("tuple(filter(true_return, [1, 2, 3, 4, 5, 6, -2, -5])) : ", tuple(filter(true_return, [1, 2, 3, 4, 5, 6, -2, -5])))
    print("list(filter(true_return, [1, 2, 3, 4, 5, 6, -2, -5]))  : ", list(filter(true_return, [1, 2, 3, 4, 5, 6, -2, -5])))
    print("list(filter(lambda x:x > 0, [1,2,3,4,5,6,-1,-2,-3,-4])) : ", list(filter(lambda x:x > 0, [1,2,3,4,5,6,-1,-2,-3,-4])))

    # hey : 16진수 , oct : 8진수, bin : 2진수,
    print("=" * 30)
    print("hey 16진수로 변환")
    print("=" * 30)
    print("hex(2546)", hex(2546))
    print("oct(2546)", oct(2546))
    print("bin(2546)", bin(2546))
    print("int(hex(2546), 16) : ", int(hex(2546), 16))
    print("int(oct(2546), 8)  : ", int(oct(2546), 8))
    print("int(bin(2546), 2)  : ", int(bin(2546), 2))

    # id 고유 주소값 리턴
    print("=" * 30)
    print("id 고유 주소값 리턴")
    print("=" * 30)

    a = 3
    print("id(3) : ", id(3))
    print("id(a) : ", id(a))

    # int(x) : 숫자를 정수 형태로 리턴하는 함수
    # int(x, radix) radix 진수로 표현된 문자열 x를 10진수로 변환
    print("=" * 30)
    print("""int(x) : 숫자를 정수 형태로 리턴하는 함수" \nint(x, radix) radix 진수로 표현된 문자열 x를 10진수로 변환""")
    print("=" * 30)
    print("int('11',2)  : ", int('11',2))
    print("int('1A',16) : ", int('1A',16))

    # isinstance(object, class) : 인스턴스, 클래스를 받는다. 인스턴스가 그 클래스의 인스턴스인지 판단하여 참거짓 리턴
    print("=" * 30)
    print("isinstance(object, class) : 인스턴스, 클래스를 받는다. 인스턴스가 그 클래스의 인스턴스인지 판단하여 참거짓 리턴")
    print("=" * 30)
    a = testClass(); b = 3;
    print("a = testClass(); b = 3")
    print("(isinstance(a, testClass)) : ", isinstance(a, testClass))
    print("(isinstance(b, testClass)) : ", isinstance(b, testClass))
    print("isinstance(b, int) : ", isinstance(b, int))

    # lambda 함수를 생성할 때 사용하는 예약어 함수를 한줄로 간결하게 만들때 사용한다
    print("=" * 30)
    print("lambda 함수를 생성할 때 사용하는 예약어 함수를 한줄로 간결하게 만들때 사용한다")
    print("람다는 def를 사용할 수 없는 곳에도 사용 가능함.")
    print("=" * 30)
    sum = lambda a,b: a+b
    print("sum(99,11) : ", sum(99,11))

    item = [lambda a,b: a+b, lambda a,b: a-b];
    print("[lambda a,b: a+b, lambda a,b: a-b] : ", item)
    print("item[0](1,2) : ", item[0](1,2))
    print("item[1](1,2) : ", item[1](1,2))

    # len : 입력값의 길이를 리턴
    print("=" * 30)
    print("len : 입력값의 길이를 리턴")
    print("=" * 30)
    print("len(len : 입력값의 길이를 리턴) : ", len("len : 입력값의 길이를 리턴"))
    print("len([1,2,3])     : ", len([1,2,3]))
    print("len((1,2,3,4))   : ", len((1,2,3,4)))

    # list(1,2,3,4) : 리스트로 만들어줌, tuple(1,2,3,4) : 튜플로 만들어줌
    print("=" * 30)
    print("list(1,2,3,4) : 리스트로 만들어줌, tuple(1,2,3,4) : 튜플로 만들어줌")
    print("=" * 30)

    print("list(1,2,3,4,5)  : ", list((1,2,3,4,5)))
    print("tuple(1,2,3,4,5) : ", tuple([1,2,3,4,5]))
    print("list('List')     : ", list('List'))
    print("list('Tuple')    : ", tuple('Tuple'))

    # map(f, iterable) : 입력받은 자료형의 각 요소가 함수 f에 의해 수행된 결과를 묶어서 리턴
    print("=" * 30)
    print("map(f, iterable) m입력받은 자료형의 각 요소가 함수 f에 의해 수행된 결과를 묶어서 리턴")
    print("=" * 30)

    print("list(map(lambda a: a*2, [1,2,3,4])) : ", list(map(lambda a: a*2, [1,2,3,4])))

    # max(iterable) , min(iterable): 자료형을 입력받아서 최대, 최소값 리턴
    print("=" * 30)
    print("max(iterable) , min(iterable): 자료형을 입력받아서 최대, 최소값 리턴")
    print("=" * 30)

    print("max([1,2,3]) : ", max([1,2,3]))
    print("min([1,2,3]) : ", min([1,2,3]))
    print("max('literable') : ", max('literable'))
    print("min('literable') : ", min('literable'))
    for i, name in enumerate("literable"):
        print(name, ord(name))

    # open(filename, [mode]) w 쓰기모드 ,r 읽기모드, a 추가모드, b 바이너리 모드
    print("=" * 30)
    print("open(filename, [mode]) w 쓰기모드 ,r 읽기모드, a 추가모드, b 바이너리 모드")
    print("=" * 30)

    try:
        f = open("etc/binary_file", "wb")

        f1 = open("etc/read_mode", "w", encoding='utf-8')
        f1.write("안녕")

        f2 = open("etc/read_mode")
        
        fa1 = open("etc/add_mode", 'a', encoding='utf-8')
        fa1.write("안녕하세용")

        #with open("etc/add_mode", 'r') as fa2:
           # print(fa2.readline())
    finally:
        f.close()
        f1.close()
        f2.close()
        fa1.close()
        pass

    # pow(x,y) : x의 y승 리턴
    print("=" * 30)
    print("pow(x,y) : x의 y승 리턴")
    print("=" * 30)

    print("pow(2,55) : ", pow(2, 55))

    # range([start,] stop [,step]) 입력받은 숫자에 해당되는 범위의 값을 반복 가능한 객체로 리턴
    print("=" * 30)
    print("range([start,] stop [,step]) 입력받은 숫자에 해당되는 범위의 값을 반복 가능한 객체로 리턴")
    print("=" * 30)

    print("list(range(5,10))        : ", list(range(5, 10)))
    print("list(range(1, 10, 2))    : ", list(range(1, 10, 2)))
    print("list(range(1, -10, -2))  : ", list(range(1, -10, -2)))


