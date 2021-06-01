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

def tongji():
    law_list = law.objects.all()
    for law_obj in law_list:
        print('law=', law_obj.law_name)
        begin = datetime.date(2019,1,1)
        end = datetime.date(2019,12,1)
        d = begin
        delta = datetime.timedelta(days=1)
        while d <= end:
            # print (d.strftime("%Y-%m-%d"))
            l=matched_law_data.objects.filter(data_type='law',keyword_name=law_obj.law_name,date=d)
            # print('len=', len(l))
            law_charts_data.objects.get_or_create(law_id=law_obj.law_id,law_name=law_obj.law_name,
                date=d,total=len(l))
            d += delta



if __name__ == '__main__':
    #统计微博热度
    tongji()
    print('done!')