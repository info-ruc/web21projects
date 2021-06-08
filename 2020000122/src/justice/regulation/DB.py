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
from regulation.models import law_charts_data
from regulation.models import explain_charts_data
from regulation.models import random_selected_data
from regulation.models import judge_law_data
from regulation.models import real_law_data
from regulation.models import judge_clause_data
from regulation.models import real_clause_data
from regulation.models import alias
from regulation.models import stopword


def delete_db():
    # law.objects.all().delete()
    # law_clause.objects.all().delete()
    # explain.objects.all().delete()
    # explain_element.objects.all().delete()
    # solr_weibo_data.objects.all().delete()
    # solr_weibo_data.objects.filter(data_type = "law").delete()
    # solr_weibo_data.objects.all().delete()
    # matched_clause_data.objects.all().delete()
    # matched_law_data.objects.all().delete()
    # law_charts_data.objects.all().delete()
    # explain_charts_data.objects.all().delete()
    # random_selected_data.objects.all().delete()
    # real_law_data.objects.all().delete()
    # real_clause_data.objects.all().delete()
    print('delete db done!')


def find_effective_law():
    law_list = law.objects.all()
    for item in law_list:
        l = multi_version_law.objects.filter(law_name = item.law_name,timeliness = "现行有效")
        if len(l) == 0:
            li = multi_version_law.objects.filter(law_name = item.law_name)
            if len(li)==0:
                print('缺失法律:',item.law_name)
            else:
                print('无现行有效: ', item.law_name)
        if len(l) >1:
            print('多个现行有效：',item.law_name)

def add_lost_law():
    law_list = law.objects.all()
    multi_version_l = multi_version_law.objects.all()
    count = len(multi_version_l)
    for item in law_list:
        l = multi_version_law.objects.filter(law_name = item.law_name)
        if len(l)==0:
            print('缺失法律:',item.law_name)
            count+=1
            multi_version_law.objects.get_or_create(
                law_id=count,
                law_name = item.law_name,
                ful_name = item.law_name,#todo
                timeliness = item.timeliness,
                department = item.department,
                efficacyLevel = item.efficacyLevel,
                releaseDate = item.releaseDate,
                effectiveDate = item.effectiveDate,
                legalCategories = item.legalCategories,
                content = item.content,
                total = item.total
            )
            clause_list = law_clause.objects.filter(law_name=item.law_name)
            for e in clause_list:
                multi_version_law_clause.objects.get_or_create(
                    law_id = count,
                    clause_id = e.clause_id,
                    law_name = item.law_name,
                    ful_name = item.law_name,#todo
                    clause_name = e.clause_name,
                    content = e.content,
                    total=e.total
                )

def modify_ful_name():
    for count in range(567,590):
        item = multi_version_law.objects.filter(law_id = count)[0]    
        clause_list = multi_version_law_clause.objects.filter(law_id = count)
        for e in clause_list:
            e.ful_name = item.ful_name
            e.save()



def delete_law():
    for count in range(567,590):
        multi_version_law.objects.filter(law_id = count).delete()



def test_date():
    begin = datetime.date(2019,6,1)
    end = datetime.date(2019,6,2)
    d = begin
    delta = datetime.timedelta(days=1)
    while d <= end:
        l=solr_weibo_data.objects.filter(doc_date=d)
        print(len(l))
        d += delta

#统计法律法条、司法解释的热度，当matched_clause_data改变后，需要重新统计
def tongji(sy,sm,sd,ey,em,ed):
    print("sy = ", sy,'type = ',type(sy))
    print("sm = ", sm,'type = ',type(sm))
    print("sd = ", sd,'type = ',type(sd))
    print("ey = ", ey,'type = ',type(ey))
    print("em = ", em,'type = ',type(em))
    print("ed = ", ed,'type = ',type(ed))

    #从法律级别匹配结果中找到的记录，需要改进为匹配函数找到的微博数据    
    law_list = law.objects.all()
    explain_list = explain.objects.all()
    # clause_list = law_clause.objects.all()
    clause_list = multi_version_law_clause.objects.all()
    element_list = explain_element.objects.all()
    
    print('==========统计法律==========')
    for law_obj in law_list:
        print('law = ', law_obj.law_name)
        l = matched_law_data.objects.filter(keyword_id = law_obj.law_id, data_type = 'law')
        num = len(l)
        law_obj.total = num
        origin_l = matched_law_data.objects.filter(keyword_id = law_obj.law_id, data_type = 'law', origin_tag=1)
        law_obj.origin_count = len(origin_l)
        law_obj.save()
    print('==========法律统计结束==========')
    

    print('==========统计司法解释==========')
    for explain_obj in explain_list:
        print('explain = ', explain_obj.explain_name)
        l = matched_law_data.objects.filter(keyword_id = explain_obj.explain_id, data_type = 'explain')
        num = len(l)
        explain_obj.total = num
        origin_l = matched_law_data.objects.filter(keyword_id = explain_obj.explain_id, data_type = 'explain', origin_tag=1)
        explain_obj.origin_count = len(origin_l)
        explain_obj.save()
    print('==========司法统计结束==========')
    
    
    print('==========统计法条==========')
    for clause_obj in clause_list:
        print('ful_name = ', clause_obj.ful_name, ' clause = ', clause_obj.clause_name)
        l = matched_clause_data.objects.filter(data_type='law', keyword_id = clause_obj.law_id, element_id = clause_obj.clause_id)
        num = len(l)
        clause_obj.total = num
        origin_l = matched_clause_data.objects.filter(data_type='law', keyword_id = clause_obj.law_id, element_id = clause_obj.clause_id, origin_tag=1)
        clause_obj.origin_count = len(origin_l)
        clause_obj.save()
    print('==========法条统计结束==========')
    
    print('==========统计司法法条==========')
    for element_obj in element_list:
        print('explain = ', element_obj.explain_name, ' element_obj = ', element_obj.element_name)
        l = matched_clause_data.objects.filter(data_type='explain', keyword_id = element_obj.explain_id, element_id=element_obj.element_id)
        element_obj.total = len(l)
        origin_l = matched_clause_data.objects.filter(data_type = 'explain', keyword_id=element_obj.explain_id, element_id=element_obj.element_id, origin_tag=1)
        element_obj.origin_count = len(origin_l)
        element_obj.save()
    print('==========统计司法法条结束==========')
    
    print('==========统计echart法律数据==========')
    law_count=0
    for law_obj in law_list:
        law_count+=1
        if law_count<128:
            continue
        print('law=', law_obj.law_name)
        begin = datetime.date(sy,sm,sd)
        end = datetime.date(ey,em,ed)
        d = begin
        delta = datetime.timedelta(days=1)
        while d < end:
            l=matched_law_data.objects.filter(data_type='law',keyword_name=law_obj.law_name,doc_date=d)
            origin_list = matched_law_data.objects.filter(data_type='law', keyword_name=law_obj.law_name,doc_date=d,origin_tag=1)
            test_list = law_charts_data.objects.filter(law_id=law_obj.law_id,law_name=law_obj.law_name,
                date=d)
            if len(test_list)>0:
                Obj = test_list[0]
                Obj.total=len(l)
                Obj.origin_count = len(origin_list)
                if len(origin_list)>0:
                    Obj.rate=len(l)/ float(len(origin_list))
                else:
                    Obj.rate=float(len(l))
                Obj.save()
            else:
                total_num = len(l)
                origin_num = len(origin_list)
                if origin_num > 0:
                    r = total_num / float(origin_num)
                else:
                    r = float(total_num)
                law_charts_data.objects.get_or_create(law_id=law_obj.law_id,law_name=law_obj.law_name,
                    date=d,total=total_num, origin_count = origin_num, rate = r)
            # print(d.strftime("%Y-%m-%d"))
            d += delta
    print('==========echart法律数据统计结束==========')
    
    print('==========统计echart司法解释数据==========')
    for explain_obj in explain_list:
        print('explain=', explain_obj.explain_name)
        begin = datetime.date(sy,sm,sd)
        end = datetime.date(ey,em,ed)
        d = begin
        delta = datetime.timedelta(days=1)
        while d < end:
            l=matched_law_data.objects.filter(data_type='explain', keyword_name=explain_obj.explain_name, doc_date=d)
            origin_list = matched_law_data.objects.filter(data_type='explain', keyword_name=explain_obj.explain_name, doc_date=d, origin_tag=1)
            test_list = explain_charts_data.objects.filter(explain_id=explain_obj.explain_id,explain_name=explain_obj.explain_name,date=d)
            #如果已存在
            if len(test_list)>0:
                Obj = test_list[0]
                Obj.total=len(l)
                Obj.origin_count = len(origin_list)
                if len(origin_list)>0:
                    Obj.rate = len(l) / float(len(origin_list))
                else:
                    Obj.rate = float(len(l))
                Obj.save()
            else:
                total_num = len(l)
                origin_num = len(origin_list)
                if origin_num > 0:
                    r = total_num / float(origin_num)
                else:
                    r = float(total_num)
                explain_charts_data.objects.get_or_create(explain_id=explain_obj.explain_id,explain_name=explain_obj.explain_name,
                    date=d,total=total_num, origin_count = origin_num, rate = r)
            # print(d.strftime("%Y-%m-%d"))
            d += delta
    print('==========echart司法数据统计结束==========')
    




#添加微博链接到两个数据库中
def add_url():
    l1=matched_clause_data.objects.all()
    l2=matched_law_data.objects.all()
    count=0
    for item in l1:
        count+=1
        if count % 1000 == 0:
            print('total = ',len(l1),'l1 count = ',count)
        temp_list=solr_weibo_data.objects.filter(weibo_id=item.weibo_id)
        item.weibo_link=temp_list[0].weibo_link
        item.save()
    count=0
    for item in l2:
        count+=1
        if count % 1000 == 0:
            print('total = ',len(l2),' l2 count = ',count)
        temp_list=solr_weibo_data.objects.filter(weibo_id=item.weibo_id)
        item.weibo_link=temp_list[0].weibo_link
        item.save()


def join():
    l=matched_clause_data.objects.all()
    print("total_len=",len(l))
    count=0
    for item in l:
        count+=1
        if count%1000==0:
            print('count=',count)
        temp_list=solr_weibo_data.objects.filter(weibo_id=item.weibo_id)
        if len(temp_list)>0:
            if len(temp_list)!=1:
                print(len(temp_list))
                print('weibo_id=',item.weibo_id)
                # for e in temp_list:
                #     print(e.author_name)
            temp=temp_list[0]
            item.user_type=temp.user_type
            item.author_name=temp.author_name
            item.tou_xiang=temp.tou_xiang
            item.doc_time=temp.doc_time
            item.doc_date=temp.doc_date
            item.doc_text=temp.doc_text
            item.weibo_source=temp.weibo_source
            item.opinion=temp.opinion
            item.month=temp.month
            item.save()




def add_alias():
    alias.objects.all().delete()
    law_list = law.objects.all()
    for item in law_list:
        if item.law_name =="中华人民共和国水法" or item.law_name =="中华人民共和国统计法":
            continue
        if item.law_name.startswith('中华人民共和国'):
            alias.objects.get_or_create(
                data_type="law",
                keyword_id=item.law_id,
                formal_name=item.law_name,
                alias_name=item.law_name[7:]
            )
    explain_list = explain.objects.all()
    for item in explain_list:
        if item.explain_name.startswith('最高人民法院'):
            alias.objects.get_or_create(
                data_type="explain",
                keyword_id=item.explain_id,
                formal_name=item.explain_name,
                alias_name=item.explain_name[6:]
            )




def select_random_weibo(num):
    print('delete ...')
    random_selected_data.objects.all().delete()
    real_law_data.objects.all().delete()
    real_clause_data.objects.all().delete()
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
            random_selected_data.objects.get_or_create(
                weibo_link = temp.weibo_link,
                weibo_id = temp.weibo_id,
                tou_xiang = temp.tou_xiang,
                author_name = temp.author_name,
                doc_text = temp.doc_text,
                weibo_source = temp.weibo_source,
                doc_date = temp.doc_date
            )











def trim_title():
    l = law.objects.all()
    for item in l:
        law_id = item.law_id
        name = item.law_name
        clauses = law_clause.objects.filter(law_id = law_id)
        for element in clauses:
            element.law_name = name
            element.save()
    l = explain.objects.all()
    for item in l:
        explain_id = item.explain_id
        name = item.explain_name
        elements = explain_element.objects.filter(explain_id = explain_id)
        for e in elements:
            e.explain_name = name
            e.save()
    l = law.objects.all()
    for item in l:
        law_id = item.law_id
        name = item.law_name
        data_list = matched_clause_data.objects.filter(keyword_id = law_id, data_type = 'law')
        for element in data_list:
            element.keyword_name = name
            element.save()
    l = explain.objects.all()
    for item in l:
        explain_id = item.explain_id
        name = item.explain_name
        data_list = matched_clause_data.objects.filter(keyword_id = explain_id, data_type = 'explain')
        for d in data_list:
            d.keyword_name = name
            d.save()
    # pass



def test():
    print('test    test !')
    # l = law_clause.objects.all()
    # for item in l:
    #     print('len = ', len(item.law_name))
    #     print('law_name =' + item.law_name+'|')
    #     new_law_name = item.law_name.strip()
    #     print('new_law_name =' + new_law_name + '|')
    #     item.law_name = new_law_name
    #     item.save()
    # l = explain_element.objects.all()
    # for item in l:
        # print('len = ', len(item.element_name))
        # print('element_name ='+item.element_name+'|')
        # new_element_name = item.element_name.strip()
        # print('new_element_name ='+new_element_name+'|')
        # item.explain_name = item.explain_name.strip()
        # item.element_name = new_element_name
        # item.save()

def add_clause_from_text(law_id,law_name, ful_name):
    multi_version_law_clause.objects.filter(law_id=law_id,law_name=law_name,ful_name=ful_name).delete()
    
    # law_clause.objects.filter(law_id=law_id,law_name=law_name).delete()
    f=open('./law_text.txt',"r")
    count = 0
    for line in f:
        if line=="\n":
            continue
        O = re.compile('(第.{1,2}节)')
        fl = O.findall(line[0:6])
        if len(fl)>0:
            continue
        Obj = re.compile('(第.{1,5}编)')
        fl = Obj.findall(line[0:6])
        if len(fl)>0:
            continue
        OBJ = re.compile('(第.{1,5}章)')
        fl = OBJ.findall(line[0:6])
        if len(fl)>0:
            continue
        if count < 100:
            obj = re.compile('(第.{1,3}条)')
            find_list = obj.findall(line[0:5])
        else:
            obj = re.compile('(第.{1,5}条)')
            find_list = obj.findall(line[0:7])
        if len(find_list)>0 and line.startswith('第'):
            # print("find_list = ",find_list)
            if find_list[0] =="第一条":
                count += 1
                clause_name = find_list[0]
                content = line + " "
            else:
                # print('law_id = ',law_id)
                # print('law_name = ',law_name)
                print('clause_id = ',count)
                print('clause_name ='+clause_name+"|")
                # print('content = ', content)
                multi_version_law_clause.objects.get_or_create(
                    law_id = law_id,
                    law_name=law_name,
                    ful_name=ful_name,
                    clause_id=count,
                    clause_name=clause_name,
                    content=content,
                    total=0
                )
                # law_clause.objects.get_or_create(
                #     law_id = law_id,
                #     law_name=law_name,
                #     clause_id=count,
                #     clause_name=clause_name,
                #     content=content,
                #     total=0
                # )
                count += 1
                clause_name = find_list[0]
                content = line + " "        
        else:
            content += line
    # print('law_id = ',law_id)
    # print('law_name = ',law_name)
    print('clause_id = ',count)
    print('clause_name ='+clause_name+"|")
    # print('content = ', content)
    multi_version_law_clause.objects.get_or_create(
        law_id = law_id,
        law_name=law_name,
        ful_name=ful_name,
        clause_id=count,
        clause_name=clause_name,
        content=content,
        total=0
    )
    # law_clause.objects.get_or_create(
    #     law_id = law_id,
    #     law_name=law_name,
    #     clause_id=count,
    #     clause_name=clause_name,
    #     content=content,
    #     total=0
    # )
    f.close()
    
def add_law(id,name,ful_name,timeliness,department,effic):
    multi_version_law.objects.get_or_create(
        law_id = id,
        law_name = name,
        ful_name = ful_name,
        timeliness = timeliness,
        department = department,
        efficacyLevel = effic,
        releaseDate = 1093622400001,
        effectiveDate = 1093622400001,
        legalCategories = "无",
        content = "无",
        total = 0
    )
    # law.objects.get_or_create(
    #     law_id = id,
    #     law_name = name,
    #     timeliness = timeliness,
    #     department = department,
    #     efficacyLevel = effic,
    #     releaseDate = 698947200000,
    #     effectiveDate = 698947200000,
    #     legalCategories = "无",
    #     content = "无",
    #     total = 0
    # )


def get_timeliness():
    l = law.objects.filter(timeliness = " 已被修改")
    print(len(l))
    for item in l:
        print(item.law_name)
    

def modify_timeliness(name):
    l = law.objects.filter(law_name = name)
    if len(l)>0:
        item = l[0]
        item.timeliness = "现行有效"
        item.save()




def multi_sort_law_and_clause():
    origin_law_list = multi_version_law.objects.all()
    for e in origin_law_list:
        e.department = e.department.strip(' ')
        e.efficacyLevel = e.efficacyLevel.strip(' ')
        e.timeliness = e.timeliness.strip(' ')
        e.save()
    law_list = multi_version_law.objects.all()
    clause_list = multi_version_law_clause.objects.all()

    law_list = sorted(law_list, key=lambda x:(x.efficacyLevel,x.department,x.ful_name))
    for item in law_list:
        print(item.law_id,item.law_name,item.ful_name)
    
    for e in clause_list:
        e.law_id = e.law_id + 1000
        e.save()

    multi_version_law.objects.all().delete()
    
    count = 0
    for item in law_list:
        count+=1
        multi_version_law.objects.get_or_create(
            law_id=count,
            law_name=item.law_name,
            ful_name=item.ful_name,
            timeliness=item.timeliness,
            department=item.department,
            efficacyLevel=item.efficacyLevel,
            releaseDate=item.releaseDate,
            effectiveDate=item.effectiveDate,
            legalCategories=item.legalCategories,
            content=item.content,
            total=item.total
        )
        l = multi_version_law_clause.objects.filter(law_id = item.law_id+1000,law_name=item.law_name)
        for e in l:
            multi_version_law_clause.objects.get_or_create(
                law_id = count,
                clause_id = e.clause_id,
                ful_name = e.ful_name,
                law_name = e.law_name,
                clause_name = e.clause_name,
                content = e.content,
                total = e.total
            )
    multi_version_law_clause.objects.filter(law_id__gt=1000).delete()
    
def move_to_law_and_clause():
    law.objects.all().delete()
    law_clause.objects.all().delete()
    valid_law_list = multi_version_law.objects.filter(timeliness = '现行有效')
    nearly_law_list = multi_version_law.objects.filter(timeliness = '尚未生效')
    count = 0
    for Law in valid_law_list:
        count += 1
        law.objects.get_or_create(
            law_id = count,
            law_name = Law.law_name,
            timeliness = Law.timeliness,
            department = Law.department,
            efficacyLevel = Law.efficacyLevel,
            releaseDate = Law.releaseDate,
            effectiveDate= Law.effectiveDate,
            legalCategories = Law.legalCategories,
            content = Law.content,
            total = Law.total
        )
        clause_list = multi_version_law_clause.objects.filter(law_id = Law.law_id)
        for e in clause_list:
            law_clause.objects.get_or_create(
                law_id = count,
                clause_id = e.clause_id,
                law_name = e.law_name,
                clause_name = e.clause_name,
                content = e.content,
                total = e.total
            )
    for Law in nearly_law_list:
        count += 1
        law.objects.get_or_create(
            law_id = count,
            law_name = Law.law_name,
            timeliness = Law.timeliness,
            department = Law.department,
            efficacyLevel = Law.efficacyLevel,
            releaseDate = Law.releaseDate,
            effectiveDate= Law.effectiveDate,
            legalCategories = Law.legalCategories,
            content = Law.content,
            total = Law.total
        )
        clause_list = multi_version_law_clause.objects.filter(law_id = Law.law_id)
        for e in clause_list:
            law_clause.objects.get_or_create(
                law_id = count,
                clause_id = e.clause_id,
                law_name = e.law_name,
                clause_name = e.clause_name,
                content = e.content,
                total = e.total
            )
    


#去除法条内容的（法宝联想：
def trim_clause_content():
    clause_list = law_clause.objects.all()
    multi_clause_list = multi_version_law_clause.objects.all()
    print('load    completed')
    for item in clause_list:
        if "（法宝联想:" in item.content:
            print('old content = ',item.content)
            item.content = item.content.split('（法宝联想:')[0]
            print('new content = ',item.content)
            item.save()
    for e in multi_clause_list:
        if "（法宝联想:" in e.content:
            print('old content = ',e.content)
            e.content = e.content.split('（法宝联想:')[0]
            print('new content = ',e.content)
            e.save()

def trim_clause_name():
    ml = multi_version_law_clause.objects.all()
    for e in ml:
        if ' ' in e.clause_name:
            print(e.clause_name)
            s=""
            for ch in e.clause_name:
                if ch !=' ':
                    s+=ch
            print('s = ',s)
            e.clause_name = s
            e.save()
    l = law_clause.objects.all()
    for item in l:
        if ' ' in item.clause_name:
            print(item.clause_name)
            s=""
            for ch in item.clause_name:
                if ch !=' ':
                    s+=ch
            print('s = ',s)
            item.clause_name = s
            item.save()

