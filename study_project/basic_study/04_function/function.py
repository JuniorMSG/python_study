"""

04. function
    04_01. 함수란 ?
    04_02. 파이썬 함수 구조
    04_03. return 사용법
    04_04. 함수 밖의 변수 변경 (global)
    04_05. lambda
"""

# 04_01. 함수란 ?
"""
    01. 함수란 ?
        입력값 (매개변수 - parameter)을 가지고 특정 기능을 수행한 후 결과값(return value)을 돌려주는것
        라고 하지만 매개변수가 없는 함수, return이 없는 함수도 있다.
    
        그런경우 특정 기능을 수행하는데 중점이 된다.
    
    02. 함수를 사용하는 이유?
        반복적으로 사용되는 로직을 하나의 덩어리로 묶는 개념
    
        DRY - Don't Repeat Yourself
        같은 일을 두 번 하지 말라
    
        같은 코드가 두 번 이상 사용될 경우에는 재사용한다, 프로젝트가 커진다면, 작은 조각으로 나누어 코드를 재사용한다.
        코드 결합도(Coupling)가 낮아저 유지보수 비용이 절감된다.
        
    03. 함수 내부의 함수를 만들어서 호출 가능
        보통 클래스에서 많이 사용되지만
        함수 내부에 함수를 만들어서 외부에서 호출 할 수 없는 함수를 만들 수 있다.
"""

# 04_02. 파이썬 함수 구조
def fn_02():
    """
    04_02. 파이썬 함수 구조
        02_01. 기본구조
            def 함수명(매개변수):
                <문장1>
                ...~
                return 결과값
        02_02. 매개변수의 갯수를 모를때
            def 함수명(*args):
                <문장1>
                ...~
                return 결과값
        02_03. 초기값 설정
            def 함수명(매개변수=값):
                <문장1>
                ...~
                return 결과값
    """
    print("\n", "=" * 5, "02. 파이썬 함수 구조. ", "=" * 5)
    print("\n", "=" * 3, "02_01. 기본구조 ", "=" * 3)

    def calc_sum(num1, num2):
        # 매개변수, return는 생략이 가능하다.
        print('calc_sum(num1, num2) :', num1, num2)
        return num1 + num2

    # 기본 호출 방법
    print('기본 호출 방법 : ', calc_sum(5, 9))

    # 매개변수 지정 호출 방법 - 가독성이 좋다.
    print('매개변수 지정 호출 방법 :', calc_sum(num1=5, num2=9))

    print("\n", "=" * 3, "02_02. 매개변수의 갯수를 모를때", "=" * 3)

    def calc_sum2(oper, *args):
        # 매개변수, return는 생략이 가능하다.
        sum_args = 0
        if oper == '*':
            sum_args = 1

        print('calc_sum2 : ', oper)
        for data in args:
            if oper == '+':
                sum_args = data + sum_args
            elif oper == '*':
                sum_args = data * sum_args
        return sum_args

    print('calc_sum2("+", 1,2,3,4,5,6,7,8,9,10) :', calc_sum2('+', 1,2,3,4,5,6,7,8,9,10))
    print('calc_sum2("*", 1,2,3,4,5,6,7,8,9,10) :', calc_sum2('*', 1,2,3,4,5,6,7,8,9,10))


    print("\n", "=" * 3, "02_03. 매개변수의 갯수를 모를때", "=" * 3)


    def calc_sum3(*args, oper='+', two_oper='*'):
        sum_args1 = 0
        sum_args2 = 0
        if oper == '*':
            sum_args1 = 1
        if two_oper == '*':
            sum_args2 = 1

        print('calc_sum2 : ', oper)
        for data in args:
            if oper == '+':
                sum_args1 = data + sum_args1
            elif oper == '*':
                sum_args1 = data * sum_args1
            elif oper == '-':
                sum_args1 = data - sum_args1

        for data in args:
            if two_oper == '+':
                sum_args2 = data + sum_args2
            elif two_oper == '*':
                sum_args2 = data * sum_args2
            elif two_oper == '-':
                sum_args2 = data - sum_args2


        return sum_args1, sum_args2

    #초기화 시키고 싶은 매개변수는 항상 제일 끝에
    print(calc_sum3(1,2,3,4,5))
    print(calc_sum3(1,2,3,4,5, oper='*', two_oper='+'))

# fn_02()

# 04_03. return 사용법
def fn_03():
    """
    04_03. return 사용법
        파이썬은 특이하게 변수에 리턴값이 여러개가 가능하다.
        03_01. return
        03_02. multi return
        03_03. 함수 종료
    """
    print("\n", "=" * 5, "03. 함수의 결과값 (return value) ", "=" * 5)
    print("\n", "=" * 3, "03_01. return", "=" * 3)


    def return_calc(num1, num2):
        return num1 + num2
    print('calc(5, 4) :', return_calc(5, 4))


    print("\n", "=" * 3, "03_02. multi return", "=" * 3)
    def multi_return_calc(num1, num2):
        # return에 여러개의 값을 줄 수 있다.
        return num1 + num2, num1 * num2, num1 - num2, num1 / num2


    plus, mul, minus, div = multi_return_calc(5, 4)
    print('calc(5, 4) :', plus, mul, minus, div)

    print("\n", "=" * 3, "03_03. 함수 종료", "=" * 3)
    def multi_return_calc(num1, num2):
        # return은 함수를 종료시킨다.
        return num1 + num2, num1 * num2, num1 - num2, num1 / num2
        print('호출 안되는 print')

    print('calc(5, 4) :', multi_return_calc(5, 4))

# fn_03()

# 04_04. 함수 밖의 변수 변경
num_out_range = 10
def fn_04():
    """
        04_04. 함수 밖의 변수 변경
            04_01. return
            04_02. global
    """
    print("\n", "=" * 5, "04. 함수 밖의 변수 변경", "=" * 5)

    print("\n", "=" * 3, "04_01. return", "=" * 3)
    num = 10
    def num_return(num):
        return num - 99

    num = num_return(num)
    print('num :', num)

    print("\n", "=" * 3, "04_02. global", "=" * 3)

    # 함수 내부이기 때문에 num은 글로벌 변수가아님.
    num = 0
    def num_global():
        global num_out_range
        num_out_range = num_out_range - 99

    num_global()
    print('num_out_range :', num_out_range)
    print("\n", "=" * 3, "04_03.", "=" * 3)

# fn_04()

#04_05. lambda
def fn_05():

    """
        04_05. lambda
            함수 생성때 사용하는 예약어 def와 동일한 역활을 한다.
            05_01. 복잡하지 않은 함수
            05_02. def 사용이 불가능한 위치에 사용
            05_03.
    """
    print("\n", "=" * 5, "05. lambda", "=" * 5)
    print("\n", "=" * 3, "05_01. 복잡하지 않은 함수", "=" * 3)
    add = lambda a, b: a+b
    sum = add(3, 4)
    print(sum)

    print("\n", "=" * 3, "05_02. def 사용이 불가능한 위치에 사용", "=" * 3)
    calc = [lambda a, b: a+b, lambda a, b: a-b, lambda a, b: a*b, lambda a, b: a/b]
    print('inner list', calc[0](4, 9))
    print('inner list', calc[1](4, 9))
    print('inner list', calc[2](4, 9))
    print('inner list', calc[3](4, 9))

    dict_lst = {"add":lambda a, b: a+b, "minus": lambda a, b: a-b, "mul": lambda a, b: a*b, "div": lambda a, b: a/b}
    print('inner dict', dict_lst['add'](9, 4))
    print('inner dict', dict_lst['minus'](9, 4))
    print('inner dict', dict_lst['mul'](9, 4))
    print('inner dict', dict_lst['div'](9, 4))




fn_05()