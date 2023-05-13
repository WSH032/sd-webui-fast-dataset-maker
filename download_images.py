# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:15:53 2023

@author: WSH
"""

import pandas as pd
import requests
import os
from tqdm import tqdm
import threading


# 获取网页的内容
URL = 'https://gelbooru.com/index.php?page=dapi&json=1&s=post&q=index&limit=10&tags=hifumi_%28blue_archive%29+'
response = requests.get(URL)

if response.status_code == 200:
    # 将JSON格式的数据规范化成数据框
    df = pd.json_normalize(response.json(), record_path=["post"])
else:
    print(f"访问 {URL} 时发生错误， 代码{response.status_code}")

# 查看结果

download_dir = os.path.join( os.getcwd(), "images" )
os.makedirs(download_dir, exist_ok=True)

def download_file(file_url, image, tags):
    r = requests.get(file_url)
    if r.status_code == 200:
        image_path = os.path.join(download_dir, image)
        try:
            with open(image_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"下载 {file_url} 时发生错误， error: {e}")
        else:
            txt_path = os.path.join(download_dir, os.path.splitext(image)[0] + ".txt")
            with open(txt_path, "w") as f:
                f.write(tags)
    else:
        print(f"访问 {file_url} 时发生错误， 代码{r.status_code}")
    return 0

threads = []
# 遍历每一行数据
for row in tqdm(df.itertuples(), miniters=1):
    # 获取网址，图片名和标签
    file_url = row.file_url
    image = row.image
    tags = row.tags
    # 创建线程并下载文件
    thread = threading.Thread(target=download_file, args=(file_url, image, tags))
    threads.append(thread)
    thread.start()
# 等待所有线程完成
for thread in threads:
    thread.join()
