
"""
    Crawling_06_imgUpload
    
    https://webhook.site/
    

"""

import requests as req


res = req.get("https://webhook.site/a74aec0a-057c-49de-a624-1b3dbf804cc5")

res = req.get("https://webhook.site/a74aec0a-057c-49de-a624-1b3dbf804cc5?name=hi")

res = req.get("https://webhook.site/a74aec0a-057c-49de-a624-1b3dbf804cc5?name=hi", headers={
    "User-Agent" : "MSG"

})

#restful API 인웹 분리, 최적화
res = req.post("https://webhook.site/a74aec0a-057c-49de-a624-1b3dbf804cc5?name=hi")
res = req.post("https://webhook.site/a74aec0a-057c-49de-a624-1b3dbf804cc5", data={"name":"sg"})

# 이미지를 HTTP로 보내는 방법
# Multipart/form-data
# 8bit - 1Byte  ASCII 아스키코드  1~255 A=65, B=66, a=97, b=98
"""
    RGB 
"""

