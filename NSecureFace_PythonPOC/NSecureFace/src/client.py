import http.client, urllib.parse
import cv2
import numpy as np
import requests

conn = http.client.HTTPConnection("localhost:9091")

r = requests.get("http://localhost:9091/status")
print(r.status_code)
print(r.content)
print("------")

img = cv2.imread("/Users/junliu/Development/Project/asiapac-hackthon-2019/NSecureFace/data/face-images/liujunju/liujunju_59.png")
ret, data = cv2.imencode(".png", img)
print(ret)
print(type(data))
print(len(data))

payload = {"image": data, "length": len(data)}
r = requests.post("http://localhost:9091/recognize", data=payload)
print(r.status_code)
print(r.content)
