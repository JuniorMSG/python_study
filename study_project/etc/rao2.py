from PIL import Image
from pytesseract import *
import os
import re
import pandas as pd

record = pd.DataFrame(columns= ['Nickname', 'Power', 'Point', '4T Kill Point', '5T kill Point', '4T Kill', '5T Kill', 'Death'])


def ocrToStr(fullPath, outTxtPath, fileName, count, lang='eng', ):
    img = Image.open(fullPath)
    txtName = os.path.join(outTxtPath, fileName.split('.')[0])

    pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR' + os.sep + 'tesseract.exe'
    width, height = img.size
    if width == 1280:
        weight = 0.8
    else :
        weight = 0.9
    try:
        power = image_to_string(img.crop((int(width)/2, int(height)/7.71, int(width)/1.61, int(height)/4.90)), config='--psm 1 -c preserve_interword_spaces=1')
        point = image_to_string(img.crop((int(width)/1.85, int(height)/4.8, int(width*0.8), int(height)/4.0)), config='--psm 1 -c preserve_interword_spaces=1')
        kill_point_4t = image_to_string(img.crop((int(width)/1.476, int(height)/1.85, int(width*0.85), int(height)/1.7)), config='--psm 1 -c preserve_interword_spaces=1')
        kill_point_5t = image_to_string(img.crop((int(width)/1.4, int(height)/1.7, int(width*0.8), int(height)/1.55)), config='--psm 1 -c preserve_interword_spaces=1')
        kill_4t = image_to_string(img.crop((int(width)/1.95, int(height)/1.85, int(width*0.6), int(height)/1.7)), config='--psm 1 -c preserve_interword_spaces=1')
        kill_5t = image_to_string(img.crop((int(width)/1.95, int(height)/1.7, int(width*0.6), int(height)/1.55)), config='--psm 1 -c preserve_interword_spaces=1')
        death = image_to_string(img.crop((int(width)/1.5, int(height)/1.25, int(width*0.8), int(height)/1.15)), config='--psm 1 -c preserve_interword_spaces=1')

        power = re.findall("\d+", power)
        power = format(int("".join(power)), ',d')

        point = re.findall("\d+", point)
        point = format(int("".join(point)), ',d')

        kill_point_4t = re.findall("\d+", kill_point_4t)
        kill_point_4t = format(int("".join(kill_point_4t)), ',d')

        kill_point_5t = re.findall("\d+", kill_point_5t)
        kill_point_5t = format(int("".join(kill_point_5t)), ',d')

        kill_4t = re.findall("\d+", kill_4t)
        kill_4t = format(int("".join(kill_4t)), ',d')

        kill_5t = re.findall("\d+", kill_5t)
        kill_5t = format(int("".join(kill_5t)), ',d')

        death = re.findall("\d+", death)
        death = format(int("".join(death)), ',d')

        print(fileName)
        print('투력 :', power)
        print('처치포인트  :', point)
        print('4티 킬포   :', kill_point_4t)
        print('5티 킬포   :   ', kill_point_5t)
        print('4티 킬     :   ', kill_4t)
        print('5티 킬     :   ', kill_5t)
        print('전사       :   ', death)

        name = re.split('.png|.jpg|.jpeg', fileName)[0]
        text = re.sub('[^a-zA-Z0-9가-힣]', ' ', name).strip()


        record.loc[count] = [text, power, point, kill_point_4t, kill_point_5t, kill_4t, kill_5t, death]

    except:
        print('img_error')

outTxtPath = "C:\\rao\\rao_img_killpoint"


for root, dirs, files in os.walk("C:\\rao\\rao_img_killpoint"):
    print(root, dirs, files)
    i = 0
    for fname in files:
        fullName = os.path.join(root, fname)
        ocrToStr(fullName, outTxtPath, fname,  i, 'kor+eng')
        i += 1

print(record)
record.to_csv('C:\\rao\\rao_killpoint.csv', mode='w', encoding='CP949')
#pyinstaller --onefile rao.py