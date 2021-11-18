
# https://pyautogui.readthedocs.io/en/latest/
# pyautoGUI

# pip install pyobjc-core mac os 전용


import pyautogui
# pip install pyautogui
# 키보드 마우스 제어 라이브러리

# 마우스 이동
pyautogui.moveTo(100, 200)
pyautogui.moveRel(200, 100)
# 마우스 클릭
pyautogui.click()

# 좌표 확인
pyautogui.position()

pyautogui.typewrite('abcd')
pyautogui.press('enter')


pyautogui.typewrite('gksrmf')

pyautogui.screenshot('result.png')

# https://pyautogui.readthedocs.io/en/latest/keyboard.html#keyboard-keys


import subprocess
import time

kakao = subprocess.Popen(['C:\Program Files (x86)\Kakao\KakaoTalk\KakaoTalk.exe'])
time.sleep(3)


# pip install opencv-python
x, y = pyautogui.locateCenterOnScreen('kakao_img1.png', confidence=0.9)

x = x + 30
pyautogui.doubleClick(x, y)

pyautogui.typewrite('vkdlTjsdmfh wkehdaptpwl qhsosmsrj goqhrh dlTspdy..')
pyautogui.press('enter')

