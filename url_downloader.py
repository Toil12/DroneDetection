#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         :2024/4/17 23:28
# @Author       :ywLi
# @File         :test.py
# @Institute    :DonghaiLab

import requests
import logging

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
def ulr_download(download_path:str,urls:list[str]):
    for url in urls:
        file_name=url.split("&")[-1].split('=')[-1]
        logging.info("downloading %s",file_name)
        download_res=requests.get(url)
        with open(rf"{download_path}\{file_name}",'wb') as f:
            f.write(download_res.content)
            logging.info("downloaded")
    logging.info("all files are downloaded")

if __name__ == '__main__':

    file_path= "10.11922sciencedb.908.txt"
    download_root=r"E:\Data\10.11922sciencedb.908"
    # print(downloade_path)
    url_read_results=read_url_file(file_path)
    ulr_download(download_root,url_read_results)


