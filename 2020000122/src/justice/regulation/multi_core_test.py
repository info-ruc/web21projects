#!/usr/bin/env python
#coding:utf-8
import json
import os
import re
import datetime
import random
import django
import concurrent.futures

from regulation.models import law
from regulation.models import law_clause
from regulation.models import multi_version_law
from regulation.models import multi_version_law_clause
from regulation.models import explain
from regulation.models import explain_element
from regulation.models import solr_weibo_data
from regulation.models import matched_law_data
from regulation.models import matched_clause_data


def f(clause_obj):
    print('ful_name = ', clause_obj.ful_name, ' clause = ', clause_obj.clause_name)
    l = matched_clause_data.objects.filter(data_type='law', keyword_id = clause_obj.law_id, element_id = clause_obj.clause_id, origin_tag=0)
    num = len(l)
    clause_obj.total = num
    origin_l = matched_law_data.objects.filter(data_type='law', keyword_id = clause_obj.law_id, element_id = clause_obj.clause_id, origin_tag=1)
    clause_obj.origin_count = len(origin_l)
    clause_obj.save()

def law_tongji(law_obj, parm):
    print('param = ',param)
    # print('law = ', law_obj.law_name)
    l = matched_law_data.objects.filter(keyword_id = law_obj.law_id, data_type = 'law', origin_tag=0)
    num = len(l)
    law_obj.total = num
    origin_l = matched_law_data.objects.filter(keyword_id = law_obj.law_id, data_type = 'law', origin_tag=1)
    law_obj.origin_count = len(origin_l)
    law_obj.save()


def explain_tongji(explain_obj):
    # print('explain = ', explain_obj.explain_name)
    l = matched_law_data.objects.filter(keyword_id = explain_obj.explain_id, data_type = 'explain', origin_tag=0)
    num = len(l)
    explain_obj.total = num
    origin_l = matched_law_data.objects.filter(keyword_id = explain_obj.explain_id, data_type = 'explain', origin_tag=1)
    explain_obj.origin_count = len(origin_l)
    explain_obj.save()


def try_multi_core():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print('law')
        law_list = law.objects.all()
        executor.map(law_tongji,law_list,1)
        # clause_list = multi_version_law_clause.objects.all()
        # executor.map(f,clause_list)

def try_multi_core_1():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print('explain')
        explain_list = explain.objects.all()
        executor.map(explain_tongji, explain_list)
