"""
    Study Basic Python grammar
    Conditional Statements : if, while

    author : MSG
"""

import sys
# sys.path.append('.')
import util.printUtil as upt
 


def con_if():

    pt = upt.print_util("True & False")
    num1 = 1 #true
    num2 = 0 #false
    lis1 = [1,2,3,4]
    tup1 = (1,2,3,4)
    str1 = "String"
    dic1 = {"name":"ms", "age":18}

    strList = []
    data = []

    strList.append('num1 and num2');        data.append(num1 and num2)
    strList.append('num1 or num2');         data.append(num1 or num2)
    strList.append('not num1');             data.append(not num1)
    strList.append('1 in lis1');            data.append(1 in lis1)
    strList.append('1 not in lis1');        data.append(1 not in lis1)
    strList.append('1 in tup1');            data.append(1 in tup1)
    strList.append('1 not in tup1');        data.append(1 not in tup1)
    strList.append('"S" in str1');          data.append("S" in str1)
    strList.append('"S" not in str1');      data.append("S" not in str1)
    strList.append('"name" in dic1');       data.append("name" in dic1)
    strList.append('"name" not in dic1');   data.append("name" not in dic1)

    pt.print_list(strList, data)

def con_if_ex01():
    print("x축을 입력하세요  : ")
    x_axis = int(input())
    print("y축을 입력하세요  : ", )
    y_axis = int(input())

    if x_axis > 0 and y_axis > 0:
        print('1사분면')
    elif x_axis < 0 and y_axis > 0:
        print("2사분면")
    elif x_axis < 0 and y_axis < 0:
        print("3사분면")
    else: print("4사분면")

    return

con_if();