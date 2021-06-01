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


def get_alias():
    fout=open('alias_list.txt',"w",encoding='utf-8')
    alias_list=[]
    l=weibodata.objects.all()
    count=0
    for wb_data in l:
        count+=1
        if count%1000==0:
            print("count=",count)
        Text = wb_data.doc_text + '\n'
        if wb_data.weibo_source != "NULL":
            Text += wb_data.weibo_source
        result_list = []
        obj = re.compile('《(.+?)》')
        find_list = obj.findall(Text)
        # print('Text=',Text)
        # print('find_list=',find_list)
        for e in find_list:
            if e not in alias_list:
                alias_list.append(e)
    for element in alias_list:
        fout.write(element)
        fout.write('\n')
    fout.close()




def minus():
    fs=open('alias_list.txt',"r",encoding='utf-8')
    fout=open('alias_name.txt','w',encoding='utf-8')
    l=[]
    for line in fs:
        word=line.strip()
        l.append(word)
    law_list=law.objects.all()
    explain_list=explain.objects.all()
    lt=[]
    for e in law_list:
        if e.law_name not in lt:
            lt.append(e.law_name)
    for e in explain_list:
        if e.explain_name not in lt:
            lt.append(e.explain_name)
    s1=set(l)
    s2=set(lt)
    c=s1-s2
    alias_l=list(c)
    for item in alias_l:
        fout.write(item)
        fout.write('\n')
    fout.close()


if __name__ == '__main__':
    # get_alias()
    minus()
    print('done!')