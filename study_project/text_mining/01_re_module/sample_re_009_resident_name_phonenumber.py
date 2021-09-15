# 이름과 전화번호가 들어있는 텍스트에서 전화번호가 맞는지 검증하여 출력하시오?
text = '''홍길동 010-1234-5678
이순신 010-3333-5417
강감찬 010-2222-4125
김유신 010-6666-7454
을지문덕 010-1111-2222'''

# [1] : 결과를 저장할 변수 선언
rst1 = []  # last result

# [2] : 텍스트내 각 개행(\n) 단위로 한줄씩 a 배열 리스트로 저장
a = text.split('\n')

# [3] : 반복을 돌면서 a 리스트내 요소를 띄어쓰기 단위로 쪼개서 별도 b 배열 리스트로 저장
for oneline in a:
        b = oneline.split(' ')
        rst2 = []
        for item in b:
                #if len(item) == 3:
                if len(item) == 13:  # --- item 길이가 13자리인건 전화번호;;
                        #print( item )
                        # 전화번호 각 자릿수가 숫자가 맞는지 체크 --> isdigit() 사용
                        # 010-1234-5678
                        #print( item[ -4: ] )
                        if item[ :3 ].isdigit() and item[ 4:8 ].isdigit() and item[ -4: ].isdigit():
                                item = item[ :3 ] +" - "+ item[ 4:8 ] +" - "+ "****"
                                rst2.append( b[0] +" : "+ item )
                        else:
                                rst2.append( b[0] +" : 이 회원의 전화번호에는 숫자가 아닌 것이 있음. " )
                                
        rst1. append( " ".join(rst2) )
                                        
# [4] : 출력
print( '\n'.join(rst1) )
        
        
   
















