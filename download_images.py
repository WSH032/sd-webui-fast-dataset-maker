# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:15:53 2023

@author: WSH
"""

#@title 下载

from urllib.parse import urlencode
from IPython.display import display, Markdown
import math
import time
from tqdm import tqdm
import pandas as pd
import requests
import os
import concurrent.futures
import hashlib

# API部分
##############################################################################################
def get_api( tags: str, limit: int=100, pid: int=0 ):
    """
    根据tags获取gelbooru的API信息
    
    tags为需要查询的tags
    limit为一次获取图片的最大限制
    pid为页数

    如果成功获取图片信息，就返回一个pandas.DataFrame
    如果不成功就返回None
    """

    base_url = 'https://gelbooru.com/index.php'
    params = {'page': 'dapi',
          'json': 1,
          's': 'post',
          'q': 'index',
          'limit': limit,
          'tags': tags,
          'pid': pid,
    }
    response = requests.get(base_url, params=params)

    try:
        response.raise_for_status()  # 检查是否是200成功访问,不是就引发异常
        # 读取JSON格式的返回信息，只取其中post部分
        response_dict = response.json()
        post_info_list = response_dict.get( 'post', [] )
        df = pd.DataFrame(post_info_list)
        # 只有确实获取到了信息，才返回一个非空的pd.Dataframe，否则返回None
        if not df.empty:
            return df
        else:
            return None
    except Exception as e:
        print(f"{e}")
        return None


# 下载部分
##############################################################################################
def cul_md5(file_path):
    """
    计算文件的 MD5 哈希值
    
    file_path为需要计算的文件路径
    
    只有成功计算了哈希值才返回，否则就返回None
    """
    try:
        with open(file_path, 'rb') as f:
            md5_hash = hashlib.md5()
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        print(f"检验 {file_path} md5时发生错误 error: {e}")
        return None

def download_file(row, download_dir, timeout=10):
    """
    下载文件和将tags写入文本
    
    row是一个类似结构体，其中应包含
        file_url ： 文件连接
        image ： 文件名字
        tags ： tags字符串
        md5 ： 文件的md5字符串
    download_dir为下载地址
    timeout为超时限制，单位为秒
    
    下载失败就返回0
    下载成功返回1
    已有重复文件返回2
    （无论是否有重复有文件，tags都会被重写一次）
    """

    try:
        # 获取网址，图片名和标签
        file_url = row.file_url
        image = row.image
        tags = row.tags
        md5 = row.md5
        # 下载图片并保存到images文件夹
        with requests.get(file_url, stream=True, timeout=timeout) as r:
            # 检查是否是200成功访问,不是就引发异常
            r.raise_for_status()

            image_path = os.path.join(download_dir, image)

            # 检查已经存在的本地文件md5是否和服务器一致，如果一致就不下载了
            is_duplicate = False
            if os.path.exists(image_path):
                if cul_md5(image_path) == md5:
                    is_duplicate = True

            # 不一致，则下载图片
            if not is_duplicate:
                with open(image_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                      if chunk:
                          f.write(chunk)

            # 不管图片是否重复，都更新tag文件
            txt_path = os.path.join(download_dir, os.path.splitext(image)[0] + ".txt")
            with open(txt_path, "w") as f:
                f.write(tags)
        time.sleep(0.1)  # 休息一下，减轻服务器压力
        return 2 if is_duplicate else 1

    except Exception as e:
        tqdm.write(f"下载 {file_url} 时发生错误, error: {e}")
        return 0

def launch_executor(post_df, download_dir: str, max_workers: int=10, timeout :int=10):
    """
    多线程下载，将post_df的每一行分给一个线程
    
    post_df为包含了下载信息的panda.DataFrame
    download_dir为下载目录
    max_workers为线程数
    timeout为下载超时时间，单位为秒
        注意这个实现是靠子函数download_file中的requests库实现
        如果其中一个线程下载超时无响应，就会引发一个错误被捕获，并返回1
    
    返回一个元组，按顺序为
    (总下载任务、 成功下载数、 存在的重复数、 下载失败数)
    """
    # 使用线程池来调度下载任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        os.makedirs(download_dir, exist_ok=True)
        # tqdm.write("开始下载")
        futures = [ executor.submit(download_file, row, download_dir, timeout) for row in post_df.itertuples() ]
        # 实时显示下载进度
        all_download_number = len(futures)
        successful_download_number = 0
        duplicate_download_number = 0
        error_download_number = 0
        for future in tqdm( concurrent.futures.as_completed(futures), total=all_download_number ):
            if future.result() == 0:
                error_download_number += 1
            elif future.result() == 1:
                successful_download_number += 1
            elif future.result() == 2:
                duplicate_download_number += 1
            else:
                tqdm.write("线程返回状态异常")
                
    download_info = ( all_download_number, successful_download_number, duplicate_download_number,  error_download_number )

    tqdm.write("下载完成")
    tqdm.write(f"下载任务： {download_info[0]} 个")
    tqdm.write(f"成功完成： {download_info[1]} 个")
    tqdm.write(f"存在重复： {download_info[2]} 个")
    tqdm.write(f"下载失败： {download_info[3]} 个")
    
    return download_info


# 主部分
##############################################################################################
def Scrape_images(tags :str,
          max_images_number: int,
          download_dir :str,
          max_workers :int=10,
          unit: int=100,
          timeout :int=10,
):
    """
    从gelbooru抓取图片，图片数量为max_images_number以unit为单位向上取
    
    tags为要抓取的tag字符串
    max_images_number为要抓取的图片数量
    download_dir为下载目录
    max_workers为下载线程
    unit为下载块单位，最小为1，最大为100
    timeout为单个图片下载超时限制，单位为秒
    
    无返回值
    """

    base_url = 'https://gelbooru.com/index.php'
    test_params = {'page': 'dapi',
            'json': 1,
            's': 'post',
            'q': 'index',
            'tags': tags,
    }
    show_params = {'page': 'post',
            's': 'list',
            'q': 'index',
            'tags': tags,
    }

    show_url = base_url + '?' + urlencode(show_params)
    display(Markdown(f"[点击检查图片是否正确]({show_url})"))
    print(f"{show_url}")

    test_response = requests.get(base_url, params=test_params)
    test_response.raise_for_status()
    test_response_dict = test_response.json()
    
    count = test_response_dict.get("@attributes",{}).get("count",0)
    limit = max(1, min(100, unit) )  # 100是gelbooru限制
    max_pid = math.floor( count / limit )
    need_pid = math.floor( (max_images_number-1) / limit )
    download_count = min(max_pid, need_pid) + 1 if count != 0 else 0
    wait_time = 3

    if download_count == 0:
        print("未发现任何图像，检查下输入的tags")
        return
    else:
        print(f"找到 {count} 张图片")
        print(f"指定下载 {max_images_number} 张, 将执行 { download_count } 轮下载")
        print(f"下载将在 {wait_time} 秒后开始")

    for t in range(wait_time) :
        print(wait_time-t)
        time.sleep(1)
        
    
    class DownloadInfoCounter(object):
        """用于下载计数的类"""
        def __init__(self):
            """ 初始化所有计数为0 """
            self.all = 0
            self.success = 0
            self.duplicate = 0
            self.error = 0
        def update(self, download_info_tuple: tuple):
            """
            输入一个四长度的元组，按顺序复制给
                全部任务、成功、重复、失败

            无返回值
            """
            if download_info_tuple:
                self.all += download_info_tuple[0]
                self.success += download_info_tuple[1]
                self.duplicate += download_info_tuple[2]
                self.error += download_info_tuple[3]
        def print(self):
            """按顺序输出相关信息"""
            print("*#" * 20)
            print("下载总结")
            print(f"下载任务： {self.all} 个")
            print(f"成功完成： {self.success} 个")
            print(f"存在重复： {self.duplicate} 个")
            print(f"下载失败： {self.error} 个")
    
    download_info_counter = DownloadInfoCounter()
    
    
    for i in range(download_count) :
        divide_str = "#" * 20
        tqdm.write(f"{divide_str}\n第 {i+1} / {download_count} 轮下载进行中:")
        post_df = get_api(tags, limit=limit, pid=i)
        if post_df is not None:
            download_info_counter.update( launch_executor(post_df, download_dir, max_workers=max_workers, timeout=timeout) )
        else:
            tqdm.write("第 {i+1} 轮下载失败")
        time.sleep(0.5)  # 休息一下，减轻压力
    
    download_info_counter.print()



"""
API  https://gelbooru.com/index.php?page=wiki&s=view&id=18780
tags  https://gelbooru.com/index.php?page=wiki&s=&s=view&id=25921
cheatsheet  https://gelbooru.com/index.php?page=wiki&s=&s=view&id=26263
"""

tags = "azusa_(blue_archive)"  #@param {type:"string"}
max_images_number = 200  #@param {type:"number"}
download_dir = "images"  #@param {type:"string"}
max_workers = 10  #@param {type:"number"}
unit = 100  #@param {type:"number"}
timeout = 7  #@param {type:"number"}

Scrape_images(tags, max_images_number, download_dir, max_workers=max_workers, unit=unit, timeout=timeout)
