#!/usr/bin/env python
#coding:utf-8
import json
import os
import re
import jieba
import math
import datetime
import random
import django
import concurrent.futures
import numpy as np
import chardet

from regulation.models import law
from regulation.models import law_clause
from regulation.models import multi_version_law
from regulation.models import multi_version_law_clause
from regulation.models import explain
from regulation.models import explain_element
from regulation.models import solr_weibo_data
from regulation.models import matched_law_data
from regulation.models import matched_clause_data
from regulation.models import law_charts_data
from regulation.models import explain_charts_data
from regulation.models import random_selected_data
from regulation.models import judge_law_data
from regulation.models import real_law_data
from regulation.models import judge_clause_data
from regulation.models import real_clause_data
from regulation.models import alias
from regulation.models import stopword
from regulation.models import nn_random_data
from regulation.models import nn_label_data
from regulation.models import nn_auto_label_data
from regulation.models import nn_fine_grain_training_data

def gen_data():
    f = open('my_0403_train_data.csv',"wb")
    title = ('sentence1\tsentence2\tLabel'+'\n').encode('utf-8')
    f.write(title)
    distinct_list = []
    writing_list = []
    real_clause_list = nn_auto_label_data.objects.filter(label=1)
    positive_list = []
    for e in real_clause_list:
        if e.weibo_id not in positive_list:
            positive_list.append(e.weibo_id)

    for weiboid in positive_list:
        temp_list = nn_auto_label_data.objects.filter(label=1,weibo_id=weiboid)
        bu_chong_list = []
        already_list = []
        if len(temp_list)==1:
            d_p_t = (temp_list[0].weibo_content,temp_list[0].content)
            p_t = (temp_list[0].weibo_content,temp_list[0].content,'1.0')
            if d_p_t not in distinct_list:
                distinct_list.append(d_p_t)
                writing_list.append(p_t)
            if temp_list[0].belong_to_law[-2:]=='解释':
                qi_yu_list = explain_element.objects.filter(explain_id = temp_list[0].law_id).exclude(element_id = temp_list[0].clause_id)
                count=0
                for item in qi_yu_list:
                    if item.content != temp_list[0].content:
                        count+=1
                        if count==2:
                            break
                        d_t = (temp_list[0].weibo_content,item.content)
                        t = (temp_list[0].weibo_content,item.content,'0.0')
                        if d_t not in distinct_list:
                            distinct_list.append(d_t)
                            writing_list.append(t)
            else:
                qi_yu_list = multi_version_law_clause.objects.filter(law_id=temp_list[0].law_id).exclude(clause_id = temp_list[0].clause_id)
                count=0
                for item in qi_yu_list:
                    if item.content != temp_list[0].content:
                        count+=1
                        if count==2:
                            break
                        d_t = (temp_list[0].weibo_content,item.content)
                        t = (temp_list[0].weibo_content,item.content,'0.0')
                        if d_t not in distinct_list:
                            distinct_list.append(d_t)
                            writing_list.append(t)
        else:
            for data in temp_list:
                d_p_t = (data.weibo_content,data.content)
                p_t = (data.weibo_content,data.content,'1.0')
                already_list.append(data.content)
                if d_p_t not in distinct_list:
                    distinct_list.append(d_p_t)
                    writing_list.append(p_t)
            for data in temp_list:
                if data.belong_to_law[-2:]=='解释':
                    find_list = explain_element.objects.filter(explain_id=data.law_id).exclude(element_id=data.clause_id)
                else:
                    find_list = multi_version_law_clause.objects.filter(law_id = data.law_id).exclude(clause_id=data.clause_id)
                if len(find_list)>0:
                    count=0
                    for item in find_list:
                        if item.content not in already_list:
                            count+=1
                            if count==2:
                                break
                            d_t = (data.weibo_content,item.content)
                            t = (data.weibo_content,item.content,'0.0')
                            if d_t not in distinct_list:
                                distinct_list.append(d_t)
                                writing_list.append(t)
    
    negative_list = nn_auto_label_data.objects.filter(label=0)
    for negative_data in negative_list:
        d_t = (negative_data.weibo_content,negative_data.content)
        t = (negative_data.weibo_content,negative_data.content,'0.0')
        if d_t not in distinct_list:
            distinct_list.append(d_t)
            writing_list.append(t)
    
    # r1 = u'[a-zA-Z0-9’!"\\\\#$%&〔〕；：\'()\\*+,-./:;<=>?\r\r\n\u3000\t\n\ue010（） @，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    r1 = u'[a-zA-Z’!"\\\\#$%&〔〕：\'()\\*+-/<=>?\r\r\n\u3000\t\n\ue010（） @★、…【】《》“”‘’！[\\]^_`{|}~]+'
    for u in writing_list:
        '''
        # print(isinstance(u[0],"gbk")," ", isinstance(u[1],"gbk"))
        y1=str.encode(u[0])
        y2=str.encode(u[1])
        q1=chardet.detect(y1)
        q2=chardet.detect(y2)
        print(q1,q2)
        '''
        sentence1 = re.sub(r1,'',u[0])
        sentence1=sentence1.encode('utf-8')
        f.write(sentence1)
        f.write('\t'.encode('utf-8'))
        sentence2 = re.sub(r1,'',u[1])
        sentence2=sentence2.encode('utf-8')
        f.write(sentence2)
        f.write('\t'.encode('utf-8'))
        my_label = u[2].encode('utf-8')
        f.write(my_label)
        f.write('\n'.encode('utf-8'))
        
    f.close()

def gen_fine_train_data():
    f = open('my_0405_fine_train_data.csv',"wb")
    title = ('sentence1\tsentence2\tLabel'+'\n').encode('utf-8')
    f.write(title)
    distinct_list = []
    writing_list = []
    real_clause_list = nn_fine_grain_training_data.objects.filter(label=1)
    positive_list = []
    for e in real_clause_list:
        if e.weibo_id not in positive_list:
            positive_list.append(e.weibo_id)

    for weiboid in positive_list:
        temp_list = nn_fine_grain_training_data.objects.filter(label=1,weibo_id=weiboid)
        bu_chong_list = []
        already_list = []
        if len(temp_list)==1:
            d_p_t = (temp_list[0].weibo_content,temp_list[0].content)
            p_t = (temp_list[0].weibo_content,temp_list[0].content,'1.0')
            if d_p_t not in distinct_list:
                distinct_list.append(d_p_t)
                writing_list.append(p_t)
            if temp_list[0].belong_to_law[-2:]=='解释':
                qi_yu_list = explain_element.objects.filter(explain_id = temp_list[0].law_id).exclude(element_id = temp_list[0].clause_id)
                count=0
                for item in qi_yu_list:
                    if item.content != temp_list[0].content:
                        count+=1
                        if count==4:
                            break
                        d_t = (temp_list[0].weibo_content,item.content)
                        t = (temp_list[0].weibo_content,item.content,'0.0')
                        if d_t not in distinct_list:
                            distinct_list.append(d_t)
                            writing_list.append(t)
            else:
                qi_yu_list = multi_version_law_clause.objects.filter(law_id=temp_list[0].law_id).exclude(clause_id = temp_list[0].clause_id)
                count=0
                for item in qi_yu_list:
                    if item.content != temp_list[0].content:
                        count+=1
                        if count==4:
                            break
                        d_t = (temp_list[0].weibo_content,item.content)
                        t = (temp_list[0].weibo_content,item.content,'0.0')
                        if d_t not in distinct_list:
                            distinct_list.append(d_t)
                            writing_list.append(t)
        else:
            for data in temp_list:
                d_p_t = (data.weibo_content,data.content)
                p_t = (data.weibo_content,data.content,'1.0')
                already_list.append(data.content)
                if d_p_t not in distinct_list:
                    distinct_list.append(d_p_t)
                    writing_list.append(p_t)
            for data in temp_list:
                if data.belong_to_law[-2:]=='解释':
                    find_list = explain_element.objects.filter(explain_id=data.law_id).exclude(element_id=data.clause_id)
                else:
                    find_list = multi_version_law_clause.objects.filter(law_id = data.law_id).exclude(clause_id=data.clause_id)
                if len(find_list)>0:
                    count=0
                    for item in find_list:
                        if item.content not in already_list:
                            count+=1
                            if count==4:
                                break
                            d_t = (data.weibo_content,item.content)
                            t = (data.weibo_content,item.content,'0.0')
                            if d_t not in distinct_list:
                                distinct_list.append(d_t)
                                writing_list.append(t)
    '''
    negative_list = nn_fine_grain_training_data.objects.filter(label=0)
    for negative_data in negative_list:
        d_t = (negative_data.weibo_content,negative_data.content)
        t = (negative_data.weibo_content,negative_data.content,'0.0')
        if d_t not in distinct_list:
            distinct_list.append(d_t)
            writing_list.append(t)
    '''
    r1 = u'[a-zA-Z0-9’!"\\\\#$%&〔〕；：\'()\\*+,-./:;<=>?\r\r\n\u3000\t\n\ue010（） @，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    # r1 = u'[a-zA-Z’!"\\\\#$%&〔〕：\'()\\*+-/<=>?\r\r\n\u3000\t\n\ue010（） @★、…【】《》“”‘’！[\\]^_`{|}~]+'
    for u in writing_list:
        '''
        # print(isinstance(u[0],"gbk")," ", isinstance(u[1],"gbk"))
        y1=str.encode(u[0])
        y2=str.encode(u[1])
        q1=chardet.detect(y1)
        q2=chardet.detect(y2)
        print(q1,q2)
        '''
        sentence1 = re.sub(r1,'',u[0])
        sentence2 = re.sub(r1,'',u[1])
        if sentence1 =='' or sentence2=='':
            continue
        sentence1=sentence1.encode('utf-8')
        f.write(sentence1)
        f.write('\t'.encode('utf-8'))
        
        sentence2=sentence2.encode('utf-8')
        f.write(sentence2)
        f.write('\t'.encode('utf-8'))
        my_label = u[2].encode('utf-8')
        f.write(my_label)
        f.write('\n'.encode('utf-8'))
        
    f.close()



def gen_test_data():
    f = open('my_0403_test_data.csv',"wb")
    title = ('sentence1\tsentence2\tLabel'+'\n').encode('utf-8')
    f.write(title)
    real_clause_list = real_clause_data.objects.all()
    for item in real_clause_list:
        data_id = item.weibo_id
        law_name = item.belong_to_law
        clause_name = item.belong_to_clause
        temp_list = random_selected_data.objects.filter(weibo_id = data_id)
        if len(temp_list)>0:
            sentence1=temp_list[0].doc_text
            if temp_list[0].weibo_source != 'NULL':
                sentence1 += temp_list[0].weibo_source
            # r1 = u'[a-zA-Z0-9’!"\\\\#$%&〔〕；：\'()\\*+,-./:;<=>?\r\r\n\u3000\t\n\ue010（） @，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
            r1 = u'[a-zA-Z’!"\\\\#$%&〔〕：\'()\\*+-/<=>?\r\r\n\u3000\t\n\ue010（） @★、…【】《》“”‘’！[\\]^_`{|}~]+'
            sentence1 = re.sub(r1,'',sentence1)
            # sentence1=sentence1.replace('\n','')
            # sentence1=sentence1.replace('\t','')
            print('sentence1 = ',sentence1)
            sentence1 = sentence1.encode('utf-8')
            
            test_list = law_clause.objects.filter(law_name = law_name, clause_name = clause_name)
            if len(test_list)>0:
                f.write(sentence1)
                f.write('\t'.encode('utf-8'))
                sentence2 = test_list[0].content
                sentence2 = re.sub(r1,'',sentence2)
                print('sentence2 = ',sentence2)
                sentence2 = sentence2.encode('utf-8')
                # sentence2 = sentence2.replace('\n','')
                # sentence2 = sentence2.replace('\t','').encode('utf-8')
                f.write(sentence2)
                f.write('\t1.0\n'.encode('utf-8'))
                re_list = law_clause.objects.filter(law_name = law_name).exclude(clause_name = clause_name)
                if len(re_list)>0:
                    if len(re_list)>1:
                        index_list = []
                        for i in range(1):
                            r = random.randint(0,len(re_list)-1)
                            sentence2 = re_list[r].content
                            sentence2 = re.sub(r1,'',sentence2)
                            print('sentence2 = ',sentence2)
                            sentence2 = sentence2.encode('utf-8')
                            # sentence2 = sentence2.replace('\n','')
                            # sentence2 = sentence2.replace('\t','').encode('utf-8')
                            f.write(sentence1)
                            f.write('\t'.encode('utf-8'))
                            f.write(sentence2)
                            f.write('\t0.0\n'.encode('utf-8'))
                    else:
                        for item in re_list:
                            sentence2 = item.content
                            sentence2 = re.sub(r1,'',sentence2)
                            print('sentence2 = ',sentence2)
                            sentence2 = sentence2.encode('utf-8')
                            f.write(sentence1)
                            f.write('\t'.encode('utf-8'))
                            f.write(sentence2)
                            f.write('\t0.0\n'.encode('utf-8'))
            else:
                print('law_name = ',law_name, ' clause_name = ',clause_name)
    f.close()





def select_data_from_solr(num):
    # print('delete ...')
    # nn_random_data.objects.all().delete()
    print('loading...')
    l = solr_weibo_data.objects.all()
    Len = len(l)
    print('Len = ',Len)
    index_list = []
    count = 0
    while count<num:
        # r = random.randint(496745,496745+Len-1)
        r = random.randint(0,Len-1)
        if r not in index_list:
            index_list.append(r)
            count+=1
    index_list = sorted(index_list)
    print('index_list = ',index_list)
    for index in index_list:
        temp_list = l[index:index+1]
        # temp_list = solr_weibo_data.objects.filter(id = index)
        if len(temp_list) >0:
            temp = temp_list[0]
            nn_random_data.objects.get_or_create(
                weibo_link = temp.weibo_link,
                weibo_id = temp.weibo_id,
                tou_xiang = temp.tou_xiang,
                author_name = temp.author_name,
                doc_text = temp.doc_text,
                weibo_source = temp.weibo_source,
                doc_date = temp.doc_date
            )

