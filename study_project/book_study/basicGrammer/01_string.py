"""
    Study Basic Python grammar
    DataType : String Type

    author : MSG
"""

def index_slicing():
    Qt1 = "12'34"
    Qt2 = '12"34'
    Qt3 = """
        Study Basic Python grammar
        double quotation marks &
        quotation marks
    """
    Qt4 = '''
        Study Basic Python grammar
        double quotation marks &
        quotation marks
    '''
    print(Qt1, Qt2)
    print(Qt3, Qt4)
    print("Qt3[0]       : " + Qt3[12])
    print("Qt3[12:17]   : " + Qt3[12:17])
    print("Qt3[17:]     : " + Qt3[17:])


def string_bulit_in_fn():
    """
        문자열 내장함수
        :return:
    """
    str = '  built In Function  '
    str2 = '_'

    print("str.count('i')  : ", str.count('i'))
    print("str.count('Fn') : ", str.count('Fn'))
    print("str.find('F')   : ", str.find('F'))
    print("str.index('l')  : ", str.index('l'))
    print("str.find('Q')   : ", str.find('Q'))
    print("str.index('Q')  :  error ")
    print("str.join(str2)  : ", str2.join(list(str.split())))
    print("str.upper()              : ", str.upper())
    print("str.lower()              : ", str.lower())
    print("str.replace('In', 'Out') : ", str.replace('In', "Out"))
    print("str.split()              : ", str.split())
    print("str.lstrip() : ", str.lstrip())
    print("str.rstrip() : ", str.rstrip())
    print("str.strip()  : ", str.strip())






def Formatting():
    Fmt_num = 3; Fmt_str = "fmt_str";
    print("I eat %d %s" % (Fmt_num, Fmt_str));
    print("Error is %d%%." % 98);

    print("I Eat {0} apples".format(3));
    print("I Eat {0} apples".format("five"));

    number = 3;
    str = "five";


    print("I Eat {0} apples. so I was sick for {1} days".format(number, str));
    print("I Eat {number} apples. so I was sick for {str} days".format(number=10, str=3));
    print("I Eat {0} apples. so I was sick for {str} days".format(10, str=3));

    print( "{0:<10}".format("HI") );
    print( "{0:>10}".format("HI") )
    print( "{0:^10}".format("HI") );

    print( "{0:=^10}".format("HI") );
    print( "{0:!<10}".format("HI") );
    print( "{0:0.4f}".format(3.141592))
    print( "{0:10.4f}".format(3.141592))
    print( "{0:^10}".format("HI") );

def Stringutil():
    print("=" * 50);
    print("StringUtil");
    print("=" * 50);
    a = "Pyhton is is simple La";
    print( a.count('is') );
    print( a.index('is') );

    a = ",";

    print(a.join('TEST'));
    print("test".upper());
    print("TEST".lower());


    print(" TEST ".lstrip());
    print(" TEST ".rstrip());
    print(" TEST ".strip());


    print(" TEST ".replace("T", "Dobule T "));

    print("TEST".split("E"));




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("=" * 50); print("Basic Python Grammer"); print("=" * 50);


    # index_slicing();
    string_bulit_in_fn()
    # Formatting()
    # Stringutil()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

