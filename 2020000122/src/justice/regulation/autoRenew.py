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

from regulation import Data
from regulation import DB
from regulation import Match

#写到文件里面的命令需要encode.('utf-8'),直接在os.system()中执行的命令不需要

def gen_cmd(data_file_path,cmd_file_path):
    print(data_file_path)
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

    year1="%04d" % (sy)
    year2="%04d" % (ey)
    month1 = "%02d" % (sm)
    month2 = "%02d" % (em)
    day1 = "%02d" % (sd)
    day2 = "%02d" % (ed)
    
    data_dir = data_file_path + year1 + month1 + day1 + "_" + year2 + month2 + day2 + "/"
    
    if os.path.exists(data_dir):
        os.removedirs(data_dir)
    mkdir_cmd = "mkdir " + data_dir
    print('mkdir cmd=',mkdir_cmd)
    os.system(mkdir_cmd)
    
    # output=open('search_cmd.txt',"wb")
    output=open(cmd_file_path,"wb")
    keyword_list=[]
    formal_name_list=[]
    law_list=law.objects.all()
    explain_list=explain.objects.all()
    alias_list=alias.objects.all()
    #先把别名加到关键词列表中
    #一部法律可以有多个别名
    for e in alias_list:
        if e.alias_name not in keyword_list:
            keyword_list.append(e.alias_name)
        if e.formal_name not in formal_name_list:
            formal_name_list.append(e.formal_name)
    #再把剩下的没有别名的加到列表中
    for e in explain_list:
        if not e.explain_name.startswith('最高人民法院'):
            keyword_list.append(e.explain_name)
            formal_name_list.append(e.explain_name)
        if e.explain_name not in formal_name_list:
            keyword_list.append(e.explain_name)
            formal_name_list.append(e.explain_name)
    
    for e in law_list:
        if not e.law_name.startswith('中华人民共和国'):
            keyword_list.append(e.law_name)
            formal_name_list.append(e.law_name)
        if e.law_name not in formal_name_list:
            keyword_list.append(e.law_name)
            formal_name_list.append(e.law_name)
    
    keyword_list = sorted(keyword_list, key=lambda x:len(x), reverse=True)
    count=0
    for name in keyword_list:
        origin_name = name
        if "(" in name:
            name="\\\\(".join(name.split('('))
        if ")" in name:
            name="\\\\)".join(name.split(')'))
        count += 1
        index = "%03d" % count
        
        p1 = "weibo-query \"\\\"".encode('utf-8')
        p2 = name.encode('utf-8')
        pp2 = origin_name.encode('utf-8')
        year1="%04d" % (sy)
        year2="%04d" % (ey)
        month1 = "%02d" % (sm)
        month2 = "%02d" % (em)
        day1 = "%02d" % (sd)
        day2 = "%02d" % (ed)

        # print('year1=',sy,'month1=' + month1,'day1='+day1)
        # print('year2=',ey,'month2=' + month2,'day2='+day2)
        p3 = "\\\" AND Time:[{y1}-{m1}-{d1}T00:00:00Z TO {y2}-{m2}-{d2}T00:00:00Z]\"".format(y1=sy,m1=month1,d1=day1,y2=ey,m2=month2,d2=day2).encode('utf-8')
        
        y1 = year1.encode('utf-8')
        y2 = year2.encode('utf-8')
        m1 = month1.encode('utf-8')
        m2 = month2.encode('utf-8')
        d1 = day1.encode('utf-8')
        d2 = day2.encode('utf-8')
        hhh = "_".encode('utf-8')
        
        duration = year1 + month1 + day1 + "_" + year2 + month2 + day2
        time_range = y1 + m1 + d1 + hhh + y2 + m2 + d2
        # print('time_range = ',time_range)

        p4 = " -f ".encode('utf-8') + data_file_path.encode('utf-8')  + time_range + "/".encode('utf-8') 
        #需要提前建好存放数据文件的目录
        # print('p4= ',p4)
        # p4 = pp4.encode('utf-8')
        p5 = ".bz2".encode('utf-8')
        id = index.encode('utf-8')
        p6 = ' -S http://183.174.229.224:9993/solr/weibo\n'.encode('utf-8')
        cmd = p1 + p2 + p3 + p4 + id + pp2 + time_range + p5 + p6
        # print(cmd)
        output.write(cmd)
    output.close()
    
    #测试，正式使用需要改
    # start_date.year=2020
    # start_date.month=3
    # start_date.day=5
    
    start_date.year=ey
    start_date.month=em
    start_date.day=ed
    start_date.save()
    
    print('gen_cmd done!')
    return duration,(sy,sm,sd,ey,em,ed)



def search(cmd_file_path,solr_shell_path):
    #bin/solr-shell @search_cmd.txt
    # search_cmd = 'cat ' + cmd_file_path + ' | ' + solr_shell_path
    search_cmd = solr_shell_path + ' @' + cmd_file_path
    os.system(search_cmd)
    print('search_done!')


def unzip_file(data_file_path,duration):
    print('unzip file begin ')
    file_dir= data_file_path + duration + '/'
    print('file_dir= ',file_dir)
    fl=os.listdir(file_dir)
    for filename in fl:
        cmd = 'bunzip2 ' + file_dir + filename
        print(cmd)
        os.system(cmd)
    print('unzip completed!')
    return file_dir



def search_result_to_db(search_result_dir):
    print('delete successfully!')
    fl = os.listdir(search_result_dir)
    print('fl = ',fl)
    #删除解压失败的文件
    for fname in fl:
        if fname.endswith(".bz2"):
            p = os.path.join(search_result_dir, fname)
            os.remove(p)
            print('delete file ', p)
    file_list = os.listdir(search_result_dir)
    for fname in file_list:
        keyword_id = int(fname[0:3])
        keyword_name = fname[3:-17]
        path = os.path.join(search_result_dir, fname)
        fsize = os.path.getsize(path)
        if fsize > 0:
            file = open(path, 'r', encoding = 'utf-8')
            print('keyword_id =', keyword_id)
            print('keyword_name ='+keyword_name+'|')
            for line in file:
                data = json.loads(line)
                weibo_id = data['Id']
                if 'DataS_user_type' in data:
                    user_type = data['DataS_user_type']
                else:
                    user_type = '未知'
                if 'Url' in data:
                    Url = data['Url']
                else:
                    Url = 'https://baidu.com'
                if 'DataS_nick_name' in data:
                    author_name = data['DataS_nick_name']
                else:
                    author_name = '未知用户'
                if 'DataS_tou_xiang' in data:
                    tou_xiang = data['DataS_tou_xiang']
                else:
                    tou_xiang = 'https://baidu.com'
                if 'Time' in data:
                    doc_time = data['Time']
                else:
                    doc_time = '150000'
                if 'Date' in data:
                    doc_date = data['Date']
                else:
                    doc_date = '2019-01-01'
                if 'Text' in data:
                    doc_text = data['Text']
                else:
                    doc_text = 'nothing'
                if 'DataS_r_weibo_content' in data:
                    weibo_source = data['DataS_r_weibo_content']
                else:
                    weibo_source = "NULL"
                if 'Opinion' in data:
                    opinion = data['Opinion']
                else:
                    opinion = 0      
                try:
                    solr_weibo_data.objects.get_or_create(
                        weibo_id = weibo_id,
                        law_process =0,
                        clause_process = 0,
                        weibo_link=Url,
                        user_type = user_type,
                        author_name = author_name, 
                        tou_xiang = tou_xiang, 
                        doc_time = doc_time, 
                        doc_date = doc_date,
                        doc_text = doc_text, 
                        weibo_source = weibo_source, 
                        opinion = opinion
                    )
                except:
                    print('weibo_id = ', weibo_id)
                    print('weibo_link = ',Url)
                    print('user_type = ',user_type)
                    print('author_name = ',author_name)
                    print('tou_xiang = ',tou_xiang)
                    print('doc_time = ', doc_time)
                    print('doc_date = ',doc_date)
                    print('doc_text = ',doc_text)
                    print('weibo_source = ',weibo_source)

    LLL=solr_weibo_data.objects.filter(law_process=0)
    print('added data length =  ',len(LLL))
    print('data to database done!')



def renew():
    # data_file_path='/home/zhan_su/renew_test_dir/project/data/'
    # cmd_file_path='/home/zhan_su/renew_test_dir/project/judicial/solr/search_cmd.txt'
    # solr_shell_path='/home/zhan_su/renew_test_dir/project/judicial/solr/bin/solr-shell'
    
    # data_file_path='/home/zhan_su/db_renew_dir_2020/project/data/'
    # cmd_file_path='/home/zhan_su/db_renew_dir_2020/project/judicial/solr/search_cmd.txt'
    # solr_shell_path='/home/zhan_su/db_renew_dir_2020/project/judicial/solr/bin/solr-shell'

    data_file_path='/home/zhan_su/test_auto_renew_running_dir/project/data/'
    cmd_file_path='/home/zhan_su/test_auto_renew_running_dir/project/judicial/solr/search_cmd.txt'
    solr_shell_path='/home/zhan_su/test_auto_renew_running_dir/project/judicial/solr/bin/solr-shell'

    duration, T =gen_cmd(data_file_path,cmd_file_path)
    search(cmd_file_path,solr_shell_path)
    
    search_result_dir = unzip_file(data_file_path,duration)
    search_result_to_db(search_result_dir)
    
    #flag==0为未匹配的数据
    #匹配法律级别的微博数据,从weibodata到matched_law_data
    Match.match_law_data()
    #匹配法条级别的微博数据,从weibodata到matched_clause_data
    Match.match_clause_data()

    #统计
    DB.tongji(T[0],T[1],T[2],T[3],T[4],T[5])
    

    print("renew completed!")

