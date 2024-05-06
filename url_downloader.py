#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         :2024/4/17 23:28
# @Author       :ywLi
# @File         :val.py
# @Institute    :DonghaiLab

import requests
import logging
import time
from requests.exceptions import RequestException

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
def read_url_file(path:str,urls=[])->list[str]:
    f=open(path,'r')
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        if line[0:5]=="Info:":
            print(f"Information of the file: {line[5:-1]}")
        else:
            urls.append(line)
    return urls
def ulr_download(download_path:str,urls:list[str],):
    for url in urls:
        file_name=url.split("&")[-1].split('=')[-1]
        logging.info("downloading %s",file_name)
        response=get_response(url)
        with open(rf"{download_path}\{file_name}", 'wb') as f:
            f.write(response.content)
            logging.info("downloaded")
    logging.info("all files are downloaded")
def get_response(url,max_retries:int=8,backoff_factor:float=0.5):
    retries=1
    while retries < max_retries:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                    return response
        except RequestException:
            pass
        delay = backoff_factor * 2 ** retries
        logging.info(f"retrying，{delay}s delayed")
        retries += 1
        time.sleep(delay)
    raise Exception("request failed and the process stops")

if __name__ == '__main__':
    file_path= "10.11922sciencedb.908.txt"
    download_root=r"E:\Data\10.11922sciencedb.908"
    url_read_results=read_url_file(file_path)
    ulr_download(download_root,url_read_results)


