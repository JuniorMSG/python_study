# hello.py
import sys
"""
Author  : MSG
DATE    : 2021-02-24
Test Project
"""
def ListType():
    print("=" * 15 +  " List Type " + "=" * 15 );
    print("=" * 15 +  " append 추가 " + "=" * 15 );
    a = [1,2,3];
    a.append(4);
    print(a);
    a.append([5,6]);
    print(a);

    print("=" * 15 +  " sort 정렬 " + "=" * 15 );

    a = [3,4,1,2];
    a.sort();
    print(a);


    print("=" * 15 +  " reverse 뒤집기 " + "=" * 15 );

    a = [3,4,1,2];
    a.reverse();
    print(a);



    print("=" * 15 +  " index 위치반환 " + "=" * 15 );

    a = [3,4,1,2];
    a.index(4);
    print(a.index(4));


    print("=" * 15 +  " insert 특정요소에 삽입  " + "=" * 15 );

    a = [3,4,1,2];
    a.insert(0, 4785);
    print(a);

    print("=" * 15 +  " remove 첫번째 나오는 x를 삭제 " + "=" * 15 );

    a = [3,4,1,2,3,1,2,3,1,2,3,1,2,3];
    a.remove(3); print(a);
    a.remove(3); print(a);
    a.remove(3); print(a);
    a.remove(3); print(a);


    print("=" * 15 +  " pop 리스트의 맨 마지막 요소를 돌려주고 그 요소는 삭제 " + "=" * 15 );

    a = [3,4,1,2];
    print(a.pop()); print(a.pop());
    print(a);
    print(a.pop());
    print(a.pop());

    print("=" * 15 +  " count " + "=" * 15 );

    a = [3,4,1,2,1,1];
    print(a.count(1));


    print("=" * 15 +  " extend a 리스트에 x 리스트 추가" + "=" * 15 );

    a = [3,4,1,2];
    a.extend([4])
    a += [5];
    print(a);



def List():
    print("111111");

    tt = [1, 2, 3, ['a', 'b', 'c'], 4, 5, 6];
    print(tt)
    print(tt[0])
    print(tt[-1])

    print(tt[0:2]);

    print(tt[0:5]);
    print(tt[3][:2]);

    a = [1,2,3]
    b = [4,5,6]

    print( a + b );

    c = a + b
    print( c );
    c[1] = ["q", "s"];
    print(c)

    c[2:3] = ["q", "s", "O", "T", "M"];
    print(c)
    del c[0];
    print(c)


def TupleType():

    print("=" * 15 +  " Tuple Type " + "=" * 15 );
    print("=" * 15 +  " List  - 값변화 가능 " + "=" * 15 );
    print("=" * 15 +  " Tuple - 값 변화 불가능" + "=" * 15 );
    t1 = ()
    t2 = (1,)
    t3 = (1,2,3)
    t4 = 1, 2, 3
    t5 = ('a', 'b', ('ab', 'cd'))

    print(t1);
    print(t2);
    print(t3);
    print(t4);
    print(t5);

    tplus = t1 + t2 + t3;
    print(tplus)
    print(tplus[2])

    print("TSlice : " , tplus[1:]);
    print("TMultiple : ", (t2*3), "1234 %s" %"%테스트", "1234 {0}".format("TEST") );
    print("TMultiple : {0} ".format((t2*3)) );



def DictionaryType():
    print("=" * 15 +  " Dictionary Type " + "=" * 15 );
    print("=" * 15 +  " List  - 값변화 가능 " + "=" * 15 );
    print("=" * 15 +  " Tuple - 값 변화 불가능" + "=" * 15 );
    print("=" * 15 +  " Dictionary - HashMap Key Value 구조" + "=" * 15 );

    dic1 = { 'name': 'MS', 'age':20, 'Gender' : 'M', 'nameList': ['M', 'S', 'G'], 'nameTuple':('M','S','G')};
    print(dic1); print(dic1['name']); print(dic1['nameList']); print(dic1['nameList'][0:2]);



    print("=" * 15 +  " Dictionary Type Add " + "=" * 15 );
    dic1[2] = "Test"; print(dic1);
    print(dic1[2]);
    del dic1[2];
    print(dic1);


def DictionaryFn():
    print("=" * 15 +  " Dictionary Function " + "=" * 15 );

    print("=" * 15 +  " 1. Key를 리스트로 만들기 " + "=" * 15 );
    dic1 = {'name': 'MS', 'age': 20, 'Gender': 'M', 'nameList': ['M', 'S', 'G'], 'nameTuple': ('M', 'S', 'G')};
    dic1_key = dic1.keys();
    print(dic1);
    print("dict_keys 객체" , dic1_key);
    print("List로 변환 ", list(dic1_key))
    

    print("=" * 15 +  " 2. Value를 리스트로 만들기 " + "=" * 15 );
    dic1 = {'name': 'MS', 'age': 20, 'Gender': 'M', 'nameList': ['M', 'S', 'G'], 'nameTuple': ('M', 'S', 'G')};
    dic1_value = dic1.values();
    print(dic1_value);
    

    print("=" * 15 +  " 3. key, value쌍 얻기 " + "=" * 15 );
    dic1 = {'name': 'MS', 'age': 20, 'Gender': 'M', 'nameList': ['M', 'S', 'G'], 'nameTuple': ('M', 'S', 'G')};
    dic1_item = dic1.items();
    print("dic1.items() : ", dic1_item);


    print("=" * 15 +  " 4. key로 value 얻기 " + "=" * 15 );
    dic1 = {'name': 'MS', 'age': 20, 'Gender': 'M', 'nameList': ['M', 'S', 'G'], 'nameTuple': ('M', 'S', 'G')};
    print("dic1['name']         : ", dic1['name']);
    print("dic1.get('name')     : ", dic1.get('name'));
    print("dic1.get('noList')   : ", dic1.get('noList', "default"));



def setType():
    print("=" * 15 +  " Set Type : 집합 자료형 " + "=" * 15 );
    s1 = set([1,2,3,4,5,6]);
    s2 = set('TEST');
    s3 = set([1,2,3,4,5,6,7,8,9,0]);

    print("=" * 15 +  " Set Type : Characteristic " + "=" * 15 );
    print("=" * 15 +  " 중복 x 순서 X " + "=" * 15 );
    print("=" * 15 +  " 사용처 : 집합을 구할때 (-_-) " + "=" * 15 );
    print(s1);
    print(s2);
    print("=" * 15 + " 교집합(Intersection " + "=" * 15);
    print("s1 & s3              : ", s1 & s3);
    print("s1.intersection(s3)  : ", s1.intersection(s3));

    print("=" * 15 + " 합집합(union) " + "=" * 15);
    print("s1 & s3      : ", s1 | s3);
    print("s1.union(s3) : ", s1.union(s3));

    print("=" * 15, " 차집합(Difference) ", "=" * 15);
    print("s1 & s3              : ", s1 | s3);
    print("s1.Difference(s3)    : ", s1.difference(s3));
    print("s1.Difference(s3)    : ", s3.difference(s1));


    print("=" * 15, " 값 1개 추가 (add) ", "=" * 15);
    s1.add(8)
    print("s1.add : ", s1);



    print("=" * 15, " 값 여러개 추가 (update) ", "=" * 15);
    s1.update([10,11,12]);
    print("s1.add : ", s1);


    print("=" * 15, " 값 삭제 (remove) ", "=" * 15);
    s1.remove(11);
    print("s1.add : ", s1);

def DatyTypeTrueFalse():
    print("=" * 15, " DatyTypeTrueFalse ", "=" * 15);
    print("bool('aaa')      : ", bool('aaa'));
    print("bool('')         : ", bool(''));
    print("bool([1,2,3])    : ", bool([1,2,3]));
    print("bool([])         : ", bool([]));
    print("bool((1,2,3))    : ", bool((1,2,3)));
    print("bool(())         : ", bool(()));
    print("bool({1,2,3})    : ", bool({'name': 'ms'}));
    print("bool({})         : ", bool({}));

    print("bool(1)          : ", bool(1));
    print("bool(0)          : ", bool(0));
    print(bool(''));

    a = [1,2,3,4];
    while a:
        print(a.pop());
    if []:
        print("TRUE");
    else:
        print("FALSE");
def variable():
    print("=" * 15, " variable ", "=" * 15);
    print("a=1, b='python' c=[1,2,3]");
    a = 1; b = 'python';   c = [1, 2, 3];
    print(a,b,c)
    print("=" * 15, " 변수는 객체가 저장된 메모리의 위치를 가르키는 레퍼런스 ", "=" * 15);
    print("=" * 15, " 파이썬의 모든 자료형은 객체  ", "=" * 15);

    a=3
    b=3
    print(a is b)
    print(sys.getrefcount(3));
    c=3
    c=4
    print(sys.getrefcount(3));

    a,b = ('python', 'test');
    print(a, b)
    (a, b) = 'python', 'test';
    print(a, b)
    [c,d] = ['TT1', 'TT2'];
    print(c, d)
    c, d = d, c
    print(c, d)
    print(sys.getrefcount(c));
    del(c);
    c = 3
    print("TT2 : ", sys.getrefcount(3));
    del(c);
    print("TT2 : ", sys.getrefcount(3));

    print("=" * 15, " copy ", "=" * 15);
    a = [1,2,3];
    b = a[:];
    a[1] = 4;
    print("a : ", a);
    print("b : ", b);


    a = [1,2,3];
    b = a.copy();
    a[1] = 4;
    print("a : ", a);
    print("b : ", b);
    print("b is a  : ", b is a );



"""
==================== DATA Type STA ====================
    1. List
    2. Tuple
    3. Dictionary
    4. Set
    5. true, false
==================== DATA Type END ====================
"""
#
# List();
# ListFn();
# TupleType();
# DictionaryType();
# DictionaryFn();
# setType();
# DatyTypeTrueFalse();
# variable()
#
#
#
"""
==================== Conditional Statements STA ====================
    1. in , not in 
==================== Conditional Statements END ====================
""";
def inNotIn():
    print("1 in [1,2,3] ", 1 in [1,2,3]);

    # prompt = """;
    #     1. Add
    #     2. Del
    #     3. List
    #     4. Quit
    #
    #     Enter number: """
    # number = 0;
    # while number != 4:
    #     print(prompt);
    #     number = int(input());

    result = [x*y for x in range(2,10) for y in range (1, 10)]

    print(result)

#inNotIn();

"""
==================== Function ====================
    1. Function
    2. global 변수는 전역변수
    3. 초기값 넣는법 fn(a, b, c=true)
==================== Function ====================
""";
name = "STUDY Global"
def fn1():

    nmae = 1234;
    if __name__ == "__main__":
        print("1번 함수")
    print("if __name__ == '__main__':은 외부에서 부르면 호출을 안해요")
    return 1;

# print("1번함수 호출 : ",fn1());


"""
==================== input ouput ====================
    1. input()
    2. global 변수는 전역변수
    3. 초기값 넣는법 fn(a, b, c=true)
==================== input ouput ====================
""";

# a = input("문자를 입력하세요 : ")
# print(a)

f = open('../새파일.txt', 'w', encoding='utf-8')
for i in range(1, 11):
    data = "%d번째 줄입니다.\n" % i
    f.write(data);
f.close()


f  = open('../새파일.txt', 'r', encoding='utf-8')
f1 = open('../새파일.txt', 'r', encoding='utf-8')
f2 = open('../새파일.txt', 'r', encoding='utf-8')

line  = f.readline();
line2 = f1.readlines();
line3 = f2.read();

print("f.readline()", line)
print("f.readlines()", line2)
print("f.read()", line3)

f.close()
f1.close()
f2.close()


f = open('../새파일.txt', 'a', encoding='utf-8')
for i in range(11, 20):
    data = "%d번째 줄입니다.\n" % i
    f.write(data);
f.close()

f  = open('../새파일.txt', 'r', encoding='utf-8')
print(f.read())

with open('../새파일.txt', 'a', encoding='utf-8') as f:
    f.write('I Love You~ ')

