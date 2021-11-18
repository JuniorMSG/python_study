# api test

import requests

res = requests.get('https://opendart.fss.or.kr/api/list.json?crtfc_key=ce7694f150a102fc0d38a3368c5d32a9102c66db')

print(res.status_code)
data = res.json()
print(data['message'])

data_list = data['list']

for x in data_list:
    print(x['corp_name'])
# ce7694f150a102fc0d38a3368c5d32a9102c66db

