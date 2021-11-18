# GUI Programming
# tkinter
# 많은 컴포넌트가 있음.
# 파이썬 기본 내장툴

from tkinter import *

class GUIT():
    def __init__(self):
        self.tkhandler = Tk()
        self.tkhandler.geometry('500x500')
        self.tkhandler.title('사행산업감독위원회 자동화 핼프봇')
        self.label_title = Label(self.tkhandler, text='안녕하세요. 핼프봇 프로토타입 모델 입니다.')
        self.label_title.pack()

        self.colc_btn = Button(self.tkhandler, text='URL 수집', width=30, command=self.event_url_colc_main)
        self.colc_btn.pack()

        self.rpa_btn = Button(self.tkhandler, text='자동 채증', width=30, command=self.event_rpa_main)
        self.rpa_btn.pack()

        self.label_url = Label(self.tkhandler, text='URL 입력창')
        self.label_url.pack()

        self.text_url = Text(self.tkhandler, width=40, height=1, relief=RIDGE, bd=1)
        self.text_url.pack()

        self.scroll = Scrollbar(self.tkhandler, orient='vertical')
        self.lbox = Listbox(self.tkhandler, yscrollcommand=self.scroll.set)
        self.scroll.config(command=self.lbox.yview)

        self.scroll.pack(side='right', fill='y')
        self.lbox.pack(side='left', fill='both', expand=True)

        self.append_log('프로그램을 시작했습니다.')

    def append_log(self, msg):
        import datetime
        now = str(datetime.datetime.now())[11:-7]
        self.lbox.insert(END, "[%s] %s"%(now, msg))
        self.lbox.update()
        self.lbox.see(END)

    def event_url_colc_main(self):
        self.cocl_handler = Tk()
        self.cocl_handler.geometry('500x500')
        self.cocl_handler.title('자동수집 헬프봇')
        self.cocl_handler = Label(self.cocl_handler, text='자동수집 핼프봇은 특정 검색어에 대해서 \n 구글, 네이버, 다음을 조회하여 결과를 자동으로 수집합니다.')
        self.cocl_handler.pack()

        self.append_log('자동수집 핼프봇 실행')

    def event_rpa_main(self):
        self.rpa_handelr = Tk()
        self.rpa_handelr.geometry('500x500')
        self.rpa_handelr.title('자동채증 헬프봇')
        self.rpa_handelr = Label(self.rpa_handelr, text='자동채증 핼프봇은 특정 URL에 대해서 \n 수집가능한 사이트인지 확인하고 수집가능한 사이트일 경우 자동으로 수집합니다.')
        self.rpa_handelr.pack()
        self.append_log('자동채증 헬프봇 실행')


    def url_handler(self):
        self.urlHandler = Tk()
        self.urlHandler.geometry('500x500')
        self.urlHandler.title('사행산업감독위원회 자동화 핼프봇')
        self.label_title = Label(self.urlHandler, text='안녕하세요. 핼프봇 프로토타입 모델 입니다.')
        self.label_title.pack()

    def run(self):
        self.tkhandler.mainloop()




g = GUIT()
g.run()
