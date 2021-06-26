"""
    클래스    - 인스턴스
    틀       - 틀로 뽑힌 물건

    객체 vs 인스턴스

    a = b(); => a는 객체이다. a는 b의 인스턴스이다. 둘다 맞는표현 .. 인스턴스는 관계 위주로 설명할때 쓰는것 .

    파이썬에선 클래스 함수가 있음\
    class TestUnit:
        def TestCase


"""
from math import gcd
from math import lcm
import study
import sys


from game_pack.game.game_python import echo2
from game_pack.game import * 

class Calc:

    def __init__(self):
        self.use = 0;

    def sum(self, num1, num2):
        self.use += 1;
        return num1 + num2

    def sub(self, num1, num2):
        self.use += 1;
        return num1 + num2

    def mul(self, num1, num2):
        self.use += 1
        return num1 * num2

    def div(self, num1, num2):
        self.use += 1
        return num1 / num2

    def remainder(self, num1, num2):
        self.use += 1
        return num1 % num2

    def gcd(self, num1, num2):
        self.use += 1
        while num2:
            num1, num2 = num2, num1 % num2
            print(num1, num2)
        return num1

    def lcm1(self, num1, num2):
        """ Least Common Multiple
        """
        self.use += 1
        lcm = (num1 * num2)
        if num1 > num2:
            big = num1
        else:
            big = num2

        while True:
            if (big % num1 == 0) and (big % num2 == 0):
                gcf = big
                break
            big += 1
            print(big)
        return lcm;
    def override(self):
        print("calc1 oveeride")
    def overloading(self, num1):
        print("동일한 이름으로 다시 구현하는게 오버라이딩 calcInheritance oveeride")


class calcInheritance(Calc):
    print("calcInheritance")
    def override(self):
        print("동일한 이름으로 다시 구현하는게 오버라이딩 calcInheritance oveeride")


calc  = Calc();
calc2 = calcInheritance();

print("sum      :", calc.sum(1, 2))
print("minus    :", calc.sub(1, 2))
print("mul      :", calc.mul(1, 2))
print("div      :", calc.div(1, 2))
print("remainder:", calc.remainder(10, 4))

print("calc     :", calc.gcd(1021, 1345))
print("calc2    :", calc2.gcd(1021, 1345))
print("LCM      :", gcd(74575475464533634444444443636273547, 34734634653454352353276272732643543))
print("LCM      :", (500 * 35014 / gcd(500, 3000)))
print("LCM      :", lcm(74575475464533634444444443636273547, 34734634653454352353276272732643543))


calc2.override();

study.fn1();
print(study.name);
# print("LCM      :", calc.LCM(182, 9))

print(sys.path);
# print("LCM      :", calc.LCM(8203, 291))

echo2();
game_python.echo();