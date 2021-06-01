#!/usr/bin/env python
#coding:utf-8
import json
import os
import re
import datetime
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "justice.settings")
if django.VERSION >= (1, 7):#自动判断版本
    django.setup()

from regulation.models import law
from regulation.models import law_clause
from regulation.models import explain
from regulation.models import explain_element
from regulation.models import weibodata
from regulation.models import matched_law_data
from regulation.models import matched_data
from regulation.models import law_charts_data
from regulation.models import explain_charts_data


def remove_duplicate():
    fout=open('qiangbimingdan.txt',"w",encoding='utf-8')
    l=weibodata.objects.all()
    id_list = []
    duplicate_list=[]
    count=0
    for item in l:
        count+=1
        if count%1000==0:
            print("count=",count)
        if item.weibo_id not in id_list:
            id_list.append(item.weibo_id)
        elif item.weibo_id not in duplicate_list:
            duplicate_list.append(item.weibo_id)
            # print("id=",item.weibo_id)
    print("len(duplicate_list)=",len(duplicate_list))
    for e in duplicate_list:
        fout.write(str(e))
        fout.write('\n')
        # L=len(weibodata.objects.filter(weibo_id=e))
        # weibodata.objects.filter(weibo_id=e)[1:L].delete()
    fout.close()




def conduct():
    f=open("qiangbimingdan.txt","r",encoding='utf-8')
    for line in f:
        index=int(line)
        l=weibodata.objects.filter(weibo_id=index)
        



if __name__ == '__main__':
    # remove_duplicate()
    conduct()
    print('done!')