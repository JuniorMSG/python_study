from PIL import Image
from pytesseract import *
import configparser
import os
import re
import numpy as np
import pandas as pd

record = pd.DataFrame(columns= ['Nickname', 'Power', 'Point', 'Maxpower', 'death', '자원', '원조', '연맹 지원 횟수'])


def ocrToStr(fullPath, outTxtPath, fileName, count, lang='eng', ):
    img = Image.open(fullPath)
    txtName = os.path.join(outTxtPath, fileName.split('.')[0])

    pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR' + os.sep + 'tesseract.exe'
    width, height = img.size
    # power = image_to_string(img.crop((960, 140, 1190, 220)), config='--psm 1 -c preserve_interword_spaces=1')
    # point = image_to_string(img.crop((1300, 120, 1920, 220)), config='--psm 1 -c preserve_interword_spaces=1')
    # maxpower = image_to_string(img.crop((1300, 300, 1900, 370)), config='--psm 1 -c preserve_interword_spaces=1')
    # death = image_to_string(img.crop((1300, 530, 1900, 590)), config='--psm 1 -c preserve_interword_spaces=1')
    # resources = image_to_string(img.crop((1300, 700, 1900, 800)), config='--psm 1 -c preserve_interword_spaces=1')
    # give_resource = image_to_string(img.crop((1300, 800, 1900, 860)), config='--psm 1 -c preserve_interword_spaces=1')
    # support_al =   image_to_string(img.crop((1300, 860, 1900, 950)), config='--psm 1 -c preserve_interword_spaces=1')

    if width == 1280:
        weight = 0.8
    else :
        weight = 0.9

    power = image_to_string(img.crop((int(width)/2, int(height)/7.71, int(width)/1.61, int(height)/4.90)), config='--psm 1 -c preserve_interword_spaces=1')
    point = image_to_string(img.crop((int(width)/1.4, int(height)/9, int(width*weight), int(height)/4.90)), config='--psm 1 -c preserve_interword_spaces=1')
    maxpower = image_to_string(img.crop((int(width)/1.476, int(height)/3.6, int(width), int(height)/2.91)), config='--psm 1 -c preserve_interword_spaces=1')
    death = image_to_string(img.crop((int(width)/1.476, int(height)/2.03, int(width), int(height)/1.83)), config='--psm 1 -c preserve_interword_spaces=1')
    resources = image_to_string(img.crop((int(width)/1.476, int(height)/1.54, int(width), int(height)/1.35)), config='--psm 1 -c preserve_interword_spaces=1')
    give_resource = image_to_string(img.crop((int(width)/1.476, int(height)/1.35, int(width), int(height)/1.25)), config='--psm 1 -c preserve_interword_spaces=1')
    support_al =   image_to_string(img.crop((int(width)/1.476, int(height)/1.25, int(width), int(height)/1.13)), config='--psm 1 -c preserve_interword_spaces=1')


    power = re.findall("\d+", power)
    power = "".join(power)

    point = re.findall("\d+", point)
    point = "".join(point)

    maxpower = re.findall("\d+", maxpower)
    maxpower = "".join(maxpower)

    death = re.findall("\d+", death)
    death = "".join(death)

    resources = re.findall("\d+", resources)
    resources = "".join(resources)

    give_resource = re.findall("\d+", give_resource)
    give_resource = "".join(give_resource)

    support_al = re.findall("\d+", support_al)
    support_al = "".join(support_al)

    print(fileName)
    print('투력 :', power)
    print('처치포인트 :', point)
    print('역대 최고 전투력:', maxpower)
    print('사망자', death)
    print('채집한 자원', resources)
    print('원조', give_resource)
    print('연맹지원 횟수', support_al)

    name = re.split('.png|.jpg|.jpeg', fileName)[0]
    text = re.sub('[^a-zA-Z0-9가-힣]', ' ', name).strip()


    record.loc[count] = [text, power, point, maxpower, death, resources, give_resource, support_al]


outTxtPath = "C:\\rao\\img"


for root, dirs, files in os.walk("C:\\rao\\img"):
    print(root, dirs, files)
    i = 0
    for fname in files:
        fullName = os.path.join(root, fname)
        ocrToStr(fullName, outTxtPath, fname,  i, 'kor+eng')
        i += 1

print(record)
record.to_csv('C:\\rao\\data.csv', mode='w', encoding='CP949')
#pyinstaller --onefile rao.py