#!/usr/bin/env python
#coding:utf-8
import json
import os
import re
import datetime
import random
import django

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
from regulation.models import timestamp


def rename_file():
    path='./data/explain/'
    for file in os.listdir(path):
        index=file[:3]
        old_name=file[3:-8]
        houzhui=file[-8:]
        # print("old_name=",old_name)
        temp_name=old_name.split("(")[0]
        # print("temp_name=",temp_name)
        if temp_name[-2:]=="解释":
            new_name=temp_name
        else:
            new_name=old_name
        if new_name!=old_name:
            print("new_name=",new_name)
            os.rename(path+index+old_name+houzhui, path+index+new_name+houzhui)


def sort_data_file():
    origin = open('law_clause.json',"r",encoding='utf-8')
    output = open('trimed_sorted_law_clause.json',"w")
    law_list = []
    l=[]
    for line in origin:
        data = json.loads(line)
        source=data['_source']
        timeliness=source['timeliness']
        department = source['department']
        releaseDate = source['releaseDate']
        effectiveDate = source['effectiveDate']
        efficacyLevel = source['efficacyLevel']
        legalCategories = source['legalCategories']
        if legalCategories == "":
            legalCategories = "无"
        ful_name=(source['title']).split("\n")[0]
        t = (ful_name,timeliness,department,releaseDate,effectiveDate,efficacyLevel,legalCategories)
        if t not in l:
            l.append(t)
            law_list.append((ful_name,line))
        else:
            print('error:',t)
    law_list=sorted(law_list,key=lambda x:x[0],reverse=False)
    for item in law_list:
        output.write(item[1])
    output.close()



def lawdata2db():
    ful_name_list = []
    filepath = 'trimed_sorted_law_clause.json'
    f = open(filepath, 'r', encoding = 'utf-8')
    law_id = 0
    count=0
    for line in f:
        count+=1
        data = json.loads(line)
        source = data['_source']
        ful_name = (source['title']).split("\n")[0]
        law_name = ful_name.split("(")[0]
        law_name = law_name.split("[")[0]
        
        if ful_name not in ful_name_list:
            ful_name_list.append(ful_name)
            law_id += 1
        else:
            print('except = '+ ful_name)
            continue
        content = source['content']
        department = source['department']
        releaseDate = source['releaseDate']
        effectiveDate = source['effectiveDate']
        timeliness = source['timeliness'].strip(' ')
        efficacyLevel = source['efficacyLevel']
        legalCategories = source['legalCategories']
        if legalCategories == "":
            legalCategories = "无"
        lawsRegulations = source['lawsRegulations']

        clause_id = 0
        for item in lawsRegulations:
            clause_id += 1
            element_content = item['element']
            element_list = element_content[1:-1].split('\n')
            clause_content = ""

            for x in element_list:
                clause_content += (x.strip(', []') + '\n')
            clause_name = element_list[0].strip(', []').split(' ')[0]
            clause_name = clause_name.strip()
            # print('law_id =', law_id )
            # print('clause_id =', clause_id)
            # print('law_name = ' + law_name)
            print('clause_name =' + clause_name+'|')
            # print('clause_content = ' + clause_content)
            multi_version_law_clause.objects.get_or_create(law_id = law_id, clause_id = clause_id, 
                law_name = law_name,ful_name=ful_name, clause_name = clause_name, content = clause_content)
        print('law_id = ', law_id)
        print('law_name = ' + law_name)
        print('ful_name = '+ful_name)
        # print('legalNumber = ', legalNumber)
        # print('timeliness = ', timeliness)
        # print('department = ', department)
        # print('efficacyLevel = ', efficacyLevel)
        # print('releaseDate = ', releaseDate)
        # print('effectiveDate = ', effectiveDate)
        # print('legalCategories = ', legalCategories)
        # print('content = ', content)
        multi_version_law.objects.get_or_create(law_id = law_id, law_name = law_name, ful_name=ful_name, timeliness = timeliness,
            department = department, efficacyLevel = efficacyLevel, releaseDate = releaseDate, 
            effectiveDate = effectiveDate, legalCategories = legalCategories, content = content)
        

    

def explaindata2db():
    explain_name_list = []
    filepath = './data/judicial_explain.json'
    f = open(filepath, 'r', encoding = 'utf-8')
    explain_id = 0
    for line in f:
        data = json.loads(line)
        source = data['_source']
        explain_name = source['title']
        explain_name = (source['title']).split("\n")[0]
        explain_name = explain_name.split("[")[0]
        temp_name = explain_name.split("(")[0]
        if temp_name[-2:]=="解释":
            explain_name = temp_name
        if explain_name not in explain_name_list:
            explain_name_list.append(explain_name)
            explain_id += 1
        else:
            continue
        content = source['content']
        
        department = source['department']
        releaseDate = source['releaseDate']
        effectiveDate = source['effectiveDate']
        timeliness = source['timeliness']
        efficacyLevel = source['efficacyLevel']
        legalCategories = source['legalCategories']
        if legalCategories == "":
            legalCategories = "NAN"
        questionExplains = source['questionExplain']
        clause_id = 0
        for item in questionExplains:
            clause_id += 1
            element_content = item['element']
            element_list = element_content[1:-1].split('\n')
            clause_content = ""

            for x in element_list:
                clause_content += (x.strip(', []') + '\n')
            clause_name = element_list[0].strip(', []').split(' ')[0]
            clause_name = clause_name.strip()
            # print('explain_id =', explain_id )
            # print('element_id =', clause_id)
            # print('explain_name = ' + explain_name)
            print('element_name =' + clause_name+'|')
            # print('element_content = ' + clause_content)
            explain_element.objects.get_or_create(explain_id = explain_id, element_id = clause_id, 
                explain_name = explain_name, element_name = clause_name, content = clause_content)
        print('explain_id = ', explain_id)
        print('explain_name = ', explain_name)
        # print('timeliness = ', timeliness)
        # print('department = ', department)
        # print('efficacyLevel = ', efficacyLevel)
        # print('releaseDate = ', releaseDate)
        # print('effectiveDate = ', effectiveDate)
        # print('legalCategories = ', legalCategories)
        # print('content = ', content)
        explain.objects.get_or_create(explain_id = explain_id, explain_name = explain_name, timeliness = timeliness,
            department = department, efficacyLevel = efficacyLevel, releaseDate = releaseDate, 
            effectiveDate = effectiveDate, legalCategories = legalCategories, content = content)




def generate_regulation_query_cmd():
    output = open('./cmd_law.txt', 'wb')
    l = law.objects.all()
    law_name_list = []
    for item in l:
        law_name = item.law_name
        law_name_list.append(law_name)
    count = 0
    for name in law_name_list:
        count += 1
        index = "%03d" % count
        

        p1 = "query \"\\\"".encode('utf-8')
        p2 = name.encode('utf-8')
        for i in range(11):
            month1 = "%02d" % (i + 1)
            month2 = "%02d" % (i + 2)
            print('month1=' + month1)
            print('month2=' + month2)
            p3 = "\\\" AND Time:[2019-{m1}-01T00:00:00Z TO 2019-{m2}-01T00:00:00Z]\"".format(m1=month1, m2=month2).encode('utf-8')
            p4 = " -f ./law/".encode('utf-8')
            p5 = ".bz2".encode('utf-8')
            id = index.encode('utf-8')
            m1 = month1.encode('utf-8')
            m2 = month2.encode('utf-8')
            p6 = '\n'.encode('utf-8')
            cmd = p1 + p2 + p3 + p4 + id + p2 + m1 + m2 + p5 + p6
            output.write(cmd)
    output.close()


def generate_explain_query_cmd():
    output = open('./cmd_explain.txt', 'wb')
    l = explain.objects.all()
    explain_name_list = []
    for item in l:
        explain_name = item.explain_name
        explain_name_list.append(explain_name)
    count = 0
    for name in explain_name_list:
        count += 1
        index = "%03d" % count
        

        p1 = "query \"\\\"".encode('utf-8')
        p2 = name.encode('utf-8')
        for i in range(11):
            month1 = "%02d" % (i + 1)
            month2 = "%02d" % (i + 2)
            print('month1=' + month1)
            print('month2=' + month2)
            p3 = "\\\" AND Time:[2019-{m1}-01T00:00:00Z TO 2019-{m2}-01T00:00:00Z]\"".format(m1=month1, m2=month2).encode('utf-8')
            p4 = " -f ./explain/".encode('utf-8')
            p5 = ".bz2".encode('utf-8')
            id = index.encode('utf-8')
            m1 = month1.encode('utf-8')
            m2 = month2.encode('utf-8')
            p6 = '\n'.encode('utf-8')
            cmd = p1 + p2 + p3 + p4 + id + p2 + m1 + m2 + p5 + p6
            output.write(cmd)
    output.close()

#生成新的查询命令文件，两个文件二合一
def gen_cmd():
    #获取现在的日期
    today = datetime.date.today()
    ey=today.year
    em=today.month
    ed=today.day
    #获取上次更新数据的日期
    date_list = timestamp.objects.all()
    if len(date_list)==0:
        print('timestamp empty!')
        return
    start_date=date_list[0]
    sy=start_date.year
    sm=start_date.month
    sd=start_date.day
    date1=sy*10000+sm*100+sd
    date2=ey*10000+em*100+ed
    if date2<=date1:
        print('date1=',date1)
        print('date2=',date2)
        print("already up to date!")
        return 
    
    output=open('/Users/suzhan/Desktop/test/judicial/justice/search_dir/search_cmd.txt',"wb")
    keyword_list=[]
    formal_name_list=[]
    law_list=law.objects.all()
    explain_list=explain.objects.all()
    alias_list=alias.objects.all()
    #先把别名加到关键词列表中
    for e in alias_list:
        if e.alias_name not in keyword_list:
            keyword_list.append(e.alias_name)
        if e.formal_name not in formal_name_list:
            formal_name_list.append(e.formal_name)
    #再把剩下的没有别名的加到列表中
    for e in explain_list:
        if e.explain_name not in formal_name_list:
            keyword_list.append(e.explain_name)
            formal_name_list.append(e.explain_name)
    
    for e in law_list:
        if e.law_name not in formal_name_list:
            keyword_list.append(e.law_name)
            formal_name_list.append(e.law_name)

    count=0
    for name in keyword_list:
        count += 1
        index = "%03d" % count
        

        p1 = "query \"\\\"".encode('utf-8')
        p2 = name.encode('utf-8')
        
        month1 = "%02d" % (sm)
        month2 = "%02d" % (em)
        day1 = "%02d" % (sd)
        day2 = "%02d" % (ed)

        print('year1=',sy,'month1=' + month1,'day1='+day1)
        print('year2=',ey,'month2=' + month2,'day2='+day2)
        p3 = "\\\" AND Time:[{y1}-{m1}-{d1}T00:00:00Z TO {y2}-{m2}-{d2}T00:00:00Z]\"".format(y1=sy,m1=month1,d1=day1,y2=ey,m2=month2,d2=day2).encode('utf-8')
        p4 = " -f ./search_data/".encode('utf-8')
        p5 = ".bz2".encode('utf-8')
        id = index.encode('utf-8')
        m1 = month1.encode('utf-8')
        m2 = month2.encode('utf-8')
        p6 = '\n'.encode('utf-8')
        cmd = p1 + p2 + p3 + p4 + id + p2 + m1 + m2 + p5 + p6
        output.write(cmd)
    output.close()
    start_date.year=2019
    start_date.month=1
    start_date.day=1
    # start_date.year=ey
    # start_date.month=em
    # start_date.day=ed
    start_date.save()
    print('gen_cmd done!')


    




def weibodata2db(flag):
    if flag==1:
        data_dir = './data/law/'
    elif flag==2:
        data_dir = './data/explain/'
    fl = os.listdir(data_dir)
    
    for fname in fl:
        keyword_id = int(fname[0:3])
        keyword_name = fname[3:-8]
        month = int(fname[-8:-6])
        path = os.path.join(data_dir, fname)
        fsize = os.path.getsize(path)
        if fsize > 0:
            file = open(path, 'r', encoding = 'utf-8')
            print('keyword_id = ', keyword_id)
            print('keyword_name = ', keyword_name)
            for line in file:
                data = json.loads(line)
                if flag==1:
                    data_type = 'law'
                elif flag==2:
                    data_type = 'explain'
                weibo_id = data['Id']
                if 'DataS_user_type' in data:
                    user_type = data['DataS_user_type']
                else:
                    user_type = '未知'
                Url = data['Url']
                author_name = data['DataS_nick_name']
                tou_xiang = data['DataS_tou_xiang']
                doc_time = data['Time']
                doc_date = data['Date']
                doc_text = data['Text']
                if 'DataS_r_weibo_content' in data:
                    weibo_source = data['DataS_r_weibo_content']
                else:
                    weibo_source = "NULL"
                if 'Opinion' in data:
                    opinion = data['Opinion']
                else:
                    opinion = 0
                # if 'DataS_r_time' in data:
                #     source_date = data['DataS_r_time']
                # else:
                #     source_date = '1970-01-01 00:00:01'
                # print('doc_time = ', doc_time)
                # print('weibo_id = ', weibo_id)
                solr_weibo_data.objects.get_or_create(weibo_link=Url,keyword_id = keyword_id, weibo_id = weibo_id, 
                    user_type = user_type,author_name = author_name, tou_xiang = tou_xiang, 
                    doc_time = doc_time, doc_date = doc_date,doc_text = doc_text, weibo_source = weibo_source, 
                    opinion = opinion, month = month,data_type = data_type)




