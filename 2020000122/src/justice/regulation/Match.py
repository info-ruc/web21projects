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
from regulation.models import nn_auto_label_data
from regulation.models import nn_fine_grain_training_data

def CleanText(text):
    link=re.compile('(http[a-zA-Z0-9*-_,~·…#%.?/&=:]+)')
    result_list = link.findall(text)
    for t in result_list:
        print('t = ',t)
        text=text.replace(t,"")
    name=re.compile('(//@.+?:)')
    result_list=name.findall(text)
    for t in result_list:
        print('t = ',t)
        text=text.replace(t,"。")
    zhanghao = re.compile('@.+? ')
    result_list=zhanghao.findall(text)
    for t in result_list:
        print('t = ',t)
        text=text.replace(t,"")
    zhuanfa=re.compile('转发微博')
    result_list=zhuanfa.findall(text)
    for t in result_list:
        print('t = ',t)
        text=text.replace(t,"")
    R = u'[’//\\\\#$%&〔〕：\'()\\*+-/<=>\r \r\n\u3000\t\n\ue010（） @★…【】“”‘’！？?![\\]^_`{|}~]+'
    text=re.sub(R,'',text)
    return text



#list = [('law',law_id,公务员法，中华人民共和国公务员法),...]
#把所有解释和法律的名字当做关键词放到敏感词列表中
def load_sensitive_word():
    sensitive_word_list = []
    explain_list =  explain.objects.all()
    explain_alias_list = alias.objects.filter(data_type="explain")
    law_list = law.objects.all()
    law_alias_list = alias.objects.filter(data_type="law")
    for item in explain_list:
        sensitive_word_list.append(('explain',item.explain_id,item.explain_name, item.explain_name))
    for item in explain_alias_list:
        sensitive_word_list.append(('explain',item.keyword_id,item.alias_name, item.formal_name))
    for item in law_list:
        sensitive_word_list.append(('law',item.law_id,item.law_name, item.law_name))
    for item in law_alias_list:
        sensitive_word_list.append(('law',item.keyword_id,item.alias_name, item.formal_name))
    sensitive_word_list = sorted(sensitive_word_list, key=lambda x:len(x[2]), reverse=True)
    return sensitive_word_list


def load_stopwords():
    stopwords = [line.strip() for line in open('clause_stopword.txt',encoding='UTF-8').readlines()]
    return stopwords



_MAPPING = (u'零', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十', u'十一', u'十二', u'十三', u'十四', u'十五', u'十六', u'十七',u'十八', u'十九')
_P0 = (u'', u'十', u'百', u'千',)
_S4 = 10 ** 4

def convert2Chinese(num):
    assert (0 <= num and num < _S4)
    if num < 20:
        return _MAPPING[num]
    else:
        lst = []
        while num >= 10:
            lst.append(num % 10)
            num = num / 10
        lst.append(num)
        c = len(lst)  # 位数
        result = u''

        for idx, val in enumerate(lst):
            val = int(val)
            if val != 0:
                result += _P0[idx] + _MAPPING[val]
                if idx < c - 1 and lst[idx + 1] == 0:
                    result += u'零'
        return result[::-1]

def cal_sentence_relative_score(s_list,frame_width):
    score=0
    if len(s_list)>frame_width:
        index=0
        while (index+frame_width)<=len(s_list):
            temp_score = sum(s_list[index:index+frame_width])/float(frame_width)
            if temp_score>score:
                score=temp_score
            index+=1
    else:
        score = sum(s_list)/float(len(s_list))
    return score


def cal_sentence_dtw_score(c_list,s_list,frame_width):
    score=0
    # print('frame_width = ',frame_width)
    m = len(c_list)
    n = len(s_list)
    # print('m = ',m,' n = ',n)
    # print('c_list = ',c_list)
    # print('s_list = ',s_list)
    match_array = np.zeros((m,n),dtype=int)
    for i in range(m):
        for j in range(n):
            match_array[i][j] = 1 if c_list[i]==s_list[j] else 0
            # print(c_list[i])
    #print(match_array)
    if frame_width == 8:
        kernel=np.array([
            [1,0.5,0,0,0,0,0,0],
            [0.5,1,0.5,0,0,0,0,0],
            [0,0.5,1,0.5,0,0,0,0],
            [0,0,0.5,1,0.5,0,0,0],
            [0,0,0,0.5,1,0.5,0,0],
            [0,0,0,0,0.5,1,0.5,0],
            [0,0,0,0,0,0.5,1,0.5],
            [0,0,0,0,0,0,0.5,1]
        ])
    elif frame_width == 4:
        kernel = np.array([
            [1,0.5,0,0],
            [0.5,1,0.5,0],
            [0,0.5,1,0.5],
            [0,0,0.5,1]
        ])
    
    score_list = []
    if m < frame_width and n < frame_width:
        d=min(m,n)
    elif m<frame_width:
        d=m
    elif n<frame_width:
        d=n
    else:
        d=frame_width
    kernel = kernel[0:d,0:d]  
    #print('sum of kernel = ',kernel.sum())
    # print('0.6d = ',d*0.6)
    i=0
    while i+d<=m:
        j=0
        while j+d<=n:
            if not np.all(match_array[i:i+d,j:j+d] == 0):
                score=np.multiply(kernel,match_array[i:i+d,j:j+d]).sum()
                score_list.append(score)
            j+=1
        i+=1
    if len(score_list)>0:
        score = max(score_list)/(0.6*d)
    # print('this score = ',score)
    return score

# def convert():
#     assert(0 <= num and num < _S4)
#     if num < 10:
#         return _MAPPING[num]
#     else:
#         lst = [ ]
#         while num >= 10:
#         lst.append(num % 10)
#         num = num / 10
#         lst.append(num)
#         c = len(lst)  # 位数
#         result = u''
    
#     for idx, val in enumerate(lst):
#       if val != 0:
#         result += _P0[idx] + _MAPPING[val]
#         if idx < c - 1 and lst[idx + 1] == 0:
#           result += u'零'
    
#     return result[::-1].replace(u'一十', u'十')


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    # try:
    #     import unicodedata  # 处理ASCii码的包
    #     for i in s:
    #         unicodedata.numeric(i)  # 把一个表示数字的字符串转换为浮点数返回的函数
    #         #return True
    #     return True
    # except (TypeError, ValueError):
    #     pass
    return False


def chinese2digits(uchars_chinese):
    common_used_numerals_tmp ={'零':0, '一':1, '二':2, '两':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10, '百':100, '千':1000, '万':10000, '亿':100000000}
    common_used_numerals= dict(zip(common_used_numerals_tmp.values(), common_used_numerals_tmp.keys())) #反转
    # print(common_used_numerals)
    total = 0
    r = 1              #表示单位：个十百千...
    for i in range(len(uchars_chinese) - 1, -1, -1):
        # print("uchars_chinese = ",uchars_chinese[i])
        # print('type of chinese =',type(uchars_chinese[i]))
        val = common_used_numerals_tmp.get(uchars_chinese[i])
        # print("val = ",val)
        if val >= 10 and i == 0:  #应对 十三 十四 十*之类
            if val > r:
                r = val
                total = total + val
            else:
                r = r * val
                #total =total + r * x
        elif val >= 10:
            if val > r:
                r = val
            else:
                r = r * val
        else:
            total = total + r * val
    return total

def get_range(l):
    if len(l)!=2:
        return 0,0
    if not l[0].isnumeric():
        return 0,0
    if not l[1].isnumeric():
        return 0,0
    if l[0].isdigit():
        begin = int(l[0])
    else:
        begin = chinese2digits(l[0])
    if l[1].isdigit():
        end = int(l[1])
    else:
        end = chinese2digits(l[1])
    return begin,end

def recheck(find_list):
    result_l = []
    for item in find_list:
        if "条至第" in item:
            l = item.split("条至第")
            # print('LLLLLLLL = ',l)
            begin, end = get_range(l)
            for num in range(begin, end+1):
                result_l.append(str(num))
        elif "条到第" in item:
            l = item.split("条到第")
            # print('LLLLLLLL = ',l)
            begin, end = get_range(l)
            for num in range(begin, end+1):
                result_l.append(str(num))
        elif "条，第" in item:
            l = item.split("条，第")
            for new_item in l:
                if new_item != "" and new_item not in result_l:
                    result_l.append(new_item)
        elif "条、第" in item:
            l = item.split("条、第")
            for new_item in l:
                if new_item != "" and new_item not in result_l:
                    result_l.append(new_item)
        elif "条。第" in item:
            l = item.split("条。第")
            for new_item in l:
                if new_item != "" and new_item not in result_l:
                    result_l.append(new_item)
        elif "、" in item:
            l = item.split("、")
            for new_item in l:
                if new_item != "" and new_item not in result_l:
                    result_l.append(new_item)
        elif "，" in item:
            l = item.split("，")
            for new_item in l:
                if new_item != "" and new_item not in result_l:
                    result_l.append(new_item)
        elif "第" in item:
            new_item = item.split("第")[1]
            if new_item != "" and new_item not in result_l:
                result_l.append(new_item)
        else:
            result_l.append(item)
    return result_l


def match_multi_version_clause(text,data_type,keyword_name,element_name):
    result_t = ()
    if data_type == 'explain':
        original_data = explain_element.objects.filter(explain_name = keyword_name, element_name = element_name)
        if len(original_data) == 1:
            result_t = ('explain', original_data[0].explain_id, original_data[0].element_id)
    else:
        original_data = multi_version_law_clause.objects.filter(law_name = keyword_name, clause_name = element_name)
        if len(original_data) > 0:
            choice_list = []
            for data in original_data:
                Law_id = data.law_id
                Clause_id = data.clause_id
                feature_list = data.feature_words.split('/')
                # print('feature_list = ',feature_list)
                score = 0
                for word in feature_list:
                    result = text.find(word)
                    if result != -1:
                        score += 1
                        # print('word = ',word)
                score = score / float(len(feature_list))
                choice_list.append((Law_id,Clause_id,score))
            choice_list = sorted(choice_list, key=lambda x:x[2],reverse = True)
            # print('choice_list = ',choice_list)
            result_t = ('law',choice_list[0][0], choice_list[0][1])
            # print('result_t = ', result_t)
    return result_t

#t=（position位置,'law',中华人民共和国公务员法）
#（position位置,'explain',...）
def match_item_in_one_area(text, t, weibo_id):
    # print("t=",t,"weibo_id =",weibo_id)
    # print('origin_txt = ',text)
    if t[1]=='others':
        return []
    result_list = []
    obj = re.compile('第(.{1,5})条')
    find_list = obj.findall(text)
    # print('find_list=',find_list)
    #如果在text中有出现第几条
    if len(find_list) >0:
        # print('----------------------------------------------------------')
        # print('text  = ',text)
        # print('word_name = ', t[2])
        # print('find_list = ', find_list)
        old_list = find_list
        find_list = recheck(find_list)
        # if find_list != old_list:
        #     print('old_list = ', old_list)
        #     print('new_list = ', find_list)
        # print('checked find_list = ', find_list)
        
        for item in find_list:
            if is_number(item) == True:
                # print('item  ='+ item+ '|')
                try:
                    mynum = int(item)
                    item = convert2Chinese(mynum)
                except:
                    print('item  = ', item)
                    print('find_list = ', find_list)
                    print('text = ', text)
                    print('t = ', t)
                    print('weibo_id = ', weibo_id)
                    continue
            #判断是否存在这个法条，以及匹配多个版本的法条，把结果放在result_list中返回
            element_name = '第' + item + '条'
            result_t = match_multi_version_clause(text,t[1],t[2],element_name)
            if result_t and result_t not in result_list:
                result_list.append(result_t)
    #如果没有明确出现第几条，就根据语义相似度匹配，这个过程只针对law，对于司法解释没有处理
    
    else:
        clause_list = multi_version_law_clause.objects.filter(law_name=t[2])
        eftime_list=[]
        m_law_list = multi_version_law.objects.filter(law_name = t[2])
        for mlaw in m_law_list:
            eftime_list.append((mlaw.ful_name,mlaw.effectiveDate))
        
        if len(clause_list)>0:
            # r1 = u'[A-Za-z0-9\!\%\[\]\,\。]'
            # r1 = u'[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+'
            r1 = ur1 = u'[a-zA-Z0-9’!"#$%&\'()\\*+,-./:;<=>?\u3000\n\ue010（） @，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
            stopwords=load_stopwords()
            text = re.sub(r1,'',text)
            # print('text = ',text)
            seg_list = jieba.lcut(text)
            # print('origin seg_list = ',seg_list)
            score_list=[]
            for index in range(len(seg_list)-1,-1,-1):
                if seg_list[index] in stopwords:
                    seg_list.pop(index)

            if len(seg_list)>0:
                for cl in clause_list:
                    effectiveTime=0
                    for eftime in eftime_list:
                        if cl.ful_name==eftime[0]:
                            effectiveTime = eftime[1]
                            break
                    feature_list = cl.feature_words.split('/')
                    # common_part = set(feature_list).intersection(set(seg_list))
                    # lc = len(common_part)
                    # if lc < 3:
                    #     continue
                    # elif lc < 6:
                    #     frame_width = 4
                    # else:
                    #     frame_width = 8
                    frame_width = 8
                    # print('len of feature  = ',len(feature_list))
                    # print('len of text = ',len(seg_list))
                    score = cal_sentence_dtw_score(feature_list,seg_list,frame_width)
                    score_list.append((cl.law_id,cl.clause_id,score,effectiveTime))
                        
                if len(score_list) > 0:
                    score_list.sort(key=lambda x:x[2],reverse=True)
                    highest_score = score_list[0][2]
                    t_list = []
                    for temp in score_list:
                        if temp[2]==highest_score:
                            t_list.append(temp)
                    t_list.sort(key=lambda x:x[3],reverse=True)
                    # print('score_list = ',score_list)
                    # print('t_list = ',t_list)
                    if t_list[0][2]>1:
                        result_t = (t[1],t_list[0][0],t_list[0][1])
                        # print('weibo_id = ',weibo_id,'score = ',t_list[0][2],'result_t = ',result_t)
                        result_list.append(result_t) 
                    else:
                        result_t = (t[1],t_list[0][0],t_list[0][1])
                        # print('no match','weibo_id = ',weibo_id,'score = ',t_list[0][2],'result_t = ',result_t)


        '''
        if len(clause_list)>0:
            r1 = u'[a-zA-Z0-9’!"#$%&\'()\\*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
            stopwords=load_stopwords()
            text=re.sub(r1,'',text)
            seg_list=jieba.lcut(text)
            temp_list=[]
            for word in seg_list:
                if word not in stopwords:
                    if word not in ['\n','\t',' ','\u3000']:
                        if word not in temp_list:
                            temp_list.append(word)
            if len(temp_list)>0:
                # print('temp_list = ',temp_list)
                score_list=[]
                for e in clause_list:
                    effectiveTime=0
                    for eftime in eftime_list:
                        if e.ful_name==eftime[0]:
                            effectiveTime = eftime[1]
                            break
                    score=0
                    clause_text = e.content
                    s_list = []
                    for word in temp_list:
                        result=clause_text.find(word)
                        if result!=-1:
                            score+=1
                            s_list.append(1)
                        else:
                            s_list.append(0)
                    frame_width = 8
                    score = cal_sentence_relative_score(s_list,frame_width)
                    # score=score/float(len(temp_list))
                    score_list.append((e.law_id,e.clause_id,score,effectiveTime))
                score_list.sort(key=lambda x:x[2],reverse=True)
                highest_score = score_list[0][2]
                t_list = []
                for temp in score_list:
                    if temp[2]==highest_score:
                        t_list.append(temp)
                t_list.sort(key=lambda x:x[3],reverse=True)
                # print('score_list = ',score_list)
                # print('t_list = ',t_list)
                if t_list[0][2]>0.6:
                    result_t = (t[1],t_list[0][0],t_list[0][1])
                    # print('result_t = ',result_t)
                    result_list.append(result_t)   
        '''
        # '''
        # if len(clause_list)>0:
        #     score_list=[]
        #     for e in clause_list:
        #         effectiveTime=0
        #         for eftime in eftime_list:
        #             if e.ful_name==eftime[0]:
        #                 effectiveTime = eftime[1]
        #                 break

        #         feature_word_list = e.feature_words.split('/')
        #         score=0
        #         print('feature_list = ',feature_word_list)
        #         for word in feature_word_list:
        #             result=text.find(word)
        #             if result!=-1:
        #                 score+=1
        #                 print('word = ',word,'score = ',score)
        #         score = score/float(len(feature_word_list))
        #         score_list.append((e.law_id,e.clause_id,score,effectiveTime))
        #     score_list.sort(key=lambda x:x[2],reverse=True)
        #     highest_score = score_list[0][2]
        #     t_list = []
        #     for temp in score_list:
        #         if temp[2]==highest_score:
        #             t_list.append(temp)
        #     t_list.sort(key=lambda x:x[3],reverse=True)
        #     # print('score_list = ',score_list)
        #     print('t_list = ',t_list)
        #     if t_list[0][2]>0.6:
        #         result_t = (t[1],t_list[0][0],t_list[0][1])
        #         result_list.append(result_t)
        # '''
    return result_list

def test_dtw():
    stopwords=load_stopwords()
    clause_list = multi_version_law_clause.objects.filter(law_id=209,clause_id=6)
    feature_words=clause_list[0].feature_words
    feature_list=feature_words.split('/')
    print('feature_list = ',feature_list)
    text = '学者是学者啊，但是只是研究某一学科某一领域的学者。如果真的要制定政策，不是应该综合考虑各个相关学科的意见吗//@发刊物--联系我：所以那些暗自发力要考上大学远走高飞的女孩子们，终于可以在高中一毕业就被摁着头嫁人了呢[微笑]转发原文：// 【#学者建议降低法定结婚年龄#：男、女性分别降至20、18周岁】法定婚龄是法律规定的最低结婚年龄。目前，婚姻家庭编草案二审稿中规定“结婚年龄，男不得早于二十二周岁，女不得早于二十周岁”，延续了婚姻法的规定。有学者认为二审稿中对法定婚龄的规定依然过高，应当适当降低，建议规定“结婚年龄，男 ?'
    r1 = u'[a-zA-Z0-9’!"#$%&\'()\\*+,-./:;<=>?\u3000\n\ue010（） @，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(r1,'',text)
    seg_list=jieba.lcut(text)
    for index in range(len(seg_list)-1,-1,-1):
        if seg_list[index] in stopwords:
            seg_list.pop(index)
    print('seg_list = ',seg_list)
    score = cal_sentence_dtw_score(feature_list,seg_list,8)
    print('score = ',score)
    



#如果temp_list中有包含关系，则删除包含的结果
def trim_list(temp_list):
    result_list = []
    result_list.append(temp_list[0])
    for index in range(1,len(temp_list)):
        r1 = temp_list[index-1][0] + len(temp_list[index-1][2])
        r = temp_list[index][0] + len(temp_list[index][2])
        if r <= r1:
            continue
        else:
            result_list.append(temp_list[index])
    return result_list



#法条级别匹配核心算法，传入文本、匹配词列表和微博ID，返回匹配结果列表result_list
#返回值result_list = (‘law’, 23(法律ID), 45(法条ID))
#返回值result_list = (‘explain’, 23(解释ID), 45(解释条ID))
def match(text, match_word_list, stopword_list,weibo_id):
    # print('text =' + text)
    temp_list = []
    s=text
    #在匹配之前把停用词用空格替换
    for e in stopword_list:
        ss=e.keyword_name
        result=s.find(ss)
        if result != -1:
            tt=" "*len(ss)
            s=s.replace(ss,tt)
    #把《》中的词找出来，去除包含匹配词的书名
    remove_list = []
    shuming_obj = re.compile('《(.+?)》')
    shuming_list = shuming_obj.findall(s)
    remove_flag = 1
    if len(shuming_list)>0:
        for shuming in shuming_list:
            for e in match_word_list:
                word = e[2]
                if word in shuming:
                    # print('word == ',word," shuming = ",shuming)
                    if len(shuming)-len(word)<3:
                        remove_flag = 0
                        # print('remove flag = ',remove_flag)
                        break
                    else:
                        remove_list.append(shuming)
                        # print('remove_list = ',remove_list)
                        break
            if remove_flag == 0:
                continue
        if len(remove_list)>0:
            # print('this remove_list  = ',remove_list)
            for remove_word in remove_list:
                Result = s.find(remove_word)
                if Result != -1:
                    s=s.replace(remove_word," "*len(remove_word))
                    temp_list.append((Result,'others',remove_word))
    text = s
    # print('text ===',text)
   
    #开始匹配关键词
    for item in match_word_list:
        word = item[2]
        result = s.find(word)
        if result != -1:
            # print(word,result)
            #（position位置,'law',中华人民共和国公务员法）
            #（position位置,'explain',...）
            temp_list.append((result, item[0], item[3]))
            t=" "*len(word)
            #先匹配长的敏感词，然后把匹配上的敏感词换成等长的1，再继续匹配短的敏感词
            s=s.replace(word,t)
    text=s
    # print('text = ',text)
 

    # print(temp_list)
    if len(temp_list) == 0:
        # print('len = 0')
        return []
    elif len(temp_list) == 1:
        # print('-----------------------------------')
        # print('len = 1')
        # print('temp_list = ', temp_list)

        # print('word = ', temp_list[0])
        # print('original text = ', text)
        # print('Text = ', text[temp_list[0][0]:])
        #如果只有一部法律就全文匹配
        result_list = match_item_in_one_area(text,temp_list[0],weibo_id)
        # print('result_list=',result_list)
        # print('-----------------------------------')
        return result_list
    else:
        #按照出现的位置从前往后排列
        old_list = sorted(temp_list, key = lambda x:x[0])
        # print('sorted keyword list = ', old_list)
        temp_list = trim_list(old_list)
        # print('trimed keyword list = ', temp_list)

        # if old_list != temp_list:
        #     print('old_list = ', old_list)
        #     print('temp_list = ', temp_list)
        # print('-----------------------------------')
        # if len(temp_list) == 2:
            # print('temp_list = ', temp_list)
            # print('text = ', text)
        # print('len = ', len(temp_list))
        # print('temp_list = ', temp_list)
        # print('text = ', text)
        result_list = []
        for index in range(len(temp_list)):
            begin = temp_list[index][0]
            if index == 0:
                begin = 0
            end = -1 if index == (len(temp_list)-1) else temp_list[index+1][0]
            # print('area = ', temp_list[index])
            # print('search area =' + text[begin:end])
            l = match_item_in_one_area(text[begin:end], temp_list[index],weibo_id)
            for item in l:
                result_list.append(item)
        # print('-----------------------------------')
        return result_list
      

#法律级别匹配核心算法
#temp_list = [('law',law_id,中华人民共和国公务员法),...]
#返回值temp_list = []
def find_law(text, match_word_list, stopword_list, weibo_id):
    # print('text =' + text)
    temp_list = []
    s=text
    #在文本中把停用词用空格替换
    for e in stopword_list:
        ss = e.keyword_name
        result=s.find(ss)
        if result != -1:
            tt=" "*len(ss)
            s=s.replace(ss,tt)
    #把《》中的词找出来，去除包含匹配词的书名
    remove_list = []
    shuming_obj = re.compile('《(.+?)》')
    shuming_list = shuming_obj.findall(s)
    # print('shuming_list = ',shuming_list)
    remove_flag = 1
    if len(shuming_list)>0:
        for shuming in shuming_list:
            for e in match_word_list:
                word = e[2]
                if word in shuming:
                    # print('word == ',word," shuming = ",shuming)
                    if len(shuming)-len(word)<3:
                        remove_flag = 0
                        break
                    else:
                        remove_list.append(shuming)
            if remove_flag == 0:
                continue
        # print('remove_list = ',remove_list)
        if len(remove_list)>0:
            for remove_word in remove_list:
                Result = s.find(remove_word)
                while Result != -1:
                    replace_word = " "*len(remove_word)
                    s=s.replace(remove_word,replace_word)
                    Result = s.find(remove_word)

    #在文本中找匹配词
    for item in match_word_list:
        word = item[2]
        result = s.find(word)
        if result != -1:
            t = (item[0],item[1],item[3])
            if t not in temp_list:
                temp_list.append(t)
            t=" "*len(word)
            s=s.replace(word,t)
    return temp_list

#匹配法律级别的微博数据
def match_law_data():    
    stopword_list = stopword.objects.all()
    sensitive_word_list = load_sensitive_word()
    # wb_data_list = solr_weibo_data.objects.filter(law_process = 0)
    wb_data_list = solr_weibo_data.objects.all()
    print('load data completed!')
    count = 0
    #对于weibodata中的每一条微博（可能有重复的）去匹配到每一部法律中
    for wb_data in wb_data_list:
        count += 1
        if count % 1000 ==0:
            print('count = ', count)
        Text = wb_data.doc_text + '\n'
        original_result_list = find_law(Text, sensitive_word_list, stopword_list, wb_data.weibo_id)
        result_list = []
        if wb_data.weibo_source != "NULL":
            Text = wb_data.weibo_source
            #从微博text和原微博text中寻找关键词
            result_list = find_law(Text, sensitive_word_list, stopword_list, wb_data.weibo_id)
        lo = len(original_result_list)
        ls = len(result_list)
        #如果微博text或原微博text中存在关键词，就加入matched_law_data数据库
        #由于原数据库weibodata存在重复的微博数据，因此匹配后的微博数据也存在重复的，
        #即可能出现一条微博匹配上多部法律
        if lo>0 or ls>0:
            # print('~~~~~~~~~~~~~~~~~')
            joint_list = list(set(original_result_list).intersection(set(result_list)))
            original_result_list = list(set(original_result_list).difference(set(joint_list)))
            source_result_list = list(set(result_list).difference(set(joint_list)))
            # print('joint_list=',joint_list)
            # print('original_result_list=',original_result_list)
            # print('source_result_list=',source_result_list)
            for item in joint_list:
                matched_law_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id,
                    keyword_id = item[1], 
                    keyword_name = item[2],
                    weibo_link = wb_data.weibo_link,
                    data_type = item[0],
                    user_type=wb_data.user_type,
                    author_name=wb_data.author_name,
                    tou_xiang=wb_data.tou_xiang,
                    doc_time=wb_data.doc_time,
                    doc_date=wb_data.doc_date, 
                    doc_text=wb_data.doc_text, 
                    weibo_source=wb_data.weibo_source,
                    opinion=wb_data.opinion,
                    origin_tag = 1,
                    source_tag = 1
                )
            for item in original_result_list:
                matched_law_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id,
                    keyword_id = item[1], 
                    keyword_name = item[2],
                    weibo_link = wb_data.weibo_link,
                    data_type = item[0],
                    user_type=wb_data.user_type,
                    author_name=wb_data.author_name,
                    tou_xiang=wb_data.tou_xiang,
                    doc_time=wb_data.doc_time,
                    doc_date=wb_data.doc_date, 
                    doc_text=wb_data.doc_text, 
                    weibo_source=wb_data.weibo_source,
                    opinion=wb_data.opinion,
                    origin_tag = 1,
                    source_tag = 0
                )
            for item in source_result_list:
                matched_law_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id,
                    keyword_id = item[1], 
                    keyword_name = item[2],
                    weibo_link = wb_data.weibo_link,
                    data_type = item[0],
                    user_type=wb_data.user_type,
                    author_name=wb_data.author_name,
                    tou_xiang=wb_data.tou_xiang,
                    doc_time=wb_data.doc_time,
                    doc_date=wb_data.doc_date, 
                    doc_text=wb_data.doc_text, 
                    weibo_source=wb_data.weibo_source,
                    opinion=wb_data.opinion,
                    origin_tag = 0,
                    source_tag = 1
                )
            # print('~~~~~~~~~~~~~~~~~')
        # wb_data.law_process=1
        # wb_data.save()
        #处理过的数据标记为1
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')



#匹配法条级别的微博数据，并且保存到matched_clause_data里面
def match_clause_data():
    stopword_list = stopword.objects.all()
    sensitive_word_list = load_sensitive_word()
    # wb_data_list = solr_weibo_data.objects.filter(clause_process=0)
    wb_data_list = solr_weibo_data.objects.all()
    print('load data completed!')
    count = 0
    for wb_data in wb_data_list:
        count += 1
        if count % 1000 ==0:
            print('count = ', count)
        Text = wb_data.doc_text + '\n'
        original_result_list = match(Text, sensitive_word_list, stopword_list, wb_data.weibo_id)
        result_list=[]
        if wb_data.weibo_source != "NULL":
            Text = wb_data.weibo_source
            result_list = match(Text, sensitive_word_list, stopword_list, wb_data.weibo_id)
        lo = len(original_result_list)
        ls = len(result_list)
        if lo>0 or ls>0:
            joint_list = list(set(original_result_list).intersection(set(result_list)))
            original_result_list = list(set(original_result_list).difference(set(joint_list)))
            source_result_list = list(set(result_list).difference(set(joint_list)))

            for item in joint_list:
                if item[0] == 'law':
                    # ref = law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    ref = multi_version_law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    keyword_name = ref[0].law_name
                    element_name = ref[0].clause_name
                else:
                    ref = explain_element.objects.filter(explain_id = item[1], element_id = item[2])
                    keyword_name = ref[0].explain_name
                    element_name = ref[0].element_name
                # print('keyword_name = ', keyword_name, 'element_name = ', element_name)
                matched_clause_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id,
                    keyword_id = item[1], 
                    element_id = item[2], 
                    keyword_name = keyword_name, 
                    element_name=element_name,
                    weibo_link = wb_data.weibo_link,
                    data_type = item[0], 
                    user_type = wb_data.user_type,
                    author_name=wb_data.author_name,
                    tou_xiang=wb_data.tou_xiang,
                    doc_time=wb_data.doc_time,
                    doc_date=wb_data.doc_date,
                    doc_text=wb_data.doc_text,    
                    weibo_source=wb_data.weibo_source,
                    opinion=wb_data.opinion,
                    origin_tag = 1,
                    source_tag = 1
                )
            for item in original_result_list:
                if item[0] == 'law':
                    # ref = law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    ref = multi_version_law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    keyword_name = ref[0].law_name
                    element_name = ref[0].clause_name
                else:
                    ref = explain_element.objects.filter(explain_id = item[1], element_id = item[2])
                    keyword_name = ref[0].explain_name
                    element_name = ref[0].element_name
                # print('keyword_name = ', keyword_name, 'element_name = ', element_name)
                matched_clause_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id,
                    keyword_id = item[1], 
                    element_id = item[2], 
                    keyword_name = keyword_name, 
                    element_name=element_name,
                    weibo_link = wb_data.weibo_link,
                    data_type = item[0], 
                    user_type = wb_data.user_type,
                    author_name=wb_data.author_name,
                    tou_xiang=wb_data.tou_xiang,
                    doc_time=wb_data.doc_time,
                    doc_date=wb_data.doc_date,
                    doc_text=wb_data.doc_text,    
                    weibo_source=wb_data.weibo_source,
                    opinion=wb_data.opinion,
                    origin_tag = 1,
                    source_tag = 0
                )
            for item in source_result_list:
                if item[0] == 'law':
                    # ref = law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    ref = multi_version_law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    keyword_name = ref[0].law_name
                    element_name = ref[0].clause_name
                else:
                    ref = explain_element.objects.filter(explain_id = item[1], element_id = item[2])
                    keyword_name = ref[0].explain_name
                    element_name = ref[0].element_name
                # print('keyword_name = ', keyword_name, 'element_name = ', element_name)
                matched_clause_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id,
                    keyword_id = item[1], 
                    element_id = item[2], 
                    keyword_name = keyword_name, 
                    element_name=element_name,
                    weibo_link = wb_data.weibo_link,
                    data_type = item[0], 
                    user_type = wb_data.user_type,
                    author_name=wb_data.author_name,
                    tou_xiang=wb_data.tou_xiang,
                    doc_time=wb_data.doc_time,
                    doc_date=wb_data.doc_date,
                    doc_text=wb_data.doc_text,    
                    weibo_source=wb_data.weibo_source,
                    opinion=wb_data.opinion,
                    origin_tag = 0,
                    source_tag = 1
                )
        # wb_data.clause_process=1
        # wb_data.save()
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')





#测试法律级别匹配的准确度
def law_test_match():
    judge_law_data.objects.all().delete()
    stopword_list=stopword.objects.all()
    #把所有解释和法律的名字当做关键词放到敏感词列表中
    sensitive_word_list = load_sensitive_word()
    wb_data_list = random_selected_data.objects.all()
    print('load test data completed!')
    count = 0
    #对于random_selected_data中的每一条微博去匹配到每一部法律中
    for wb_data in wb_data_list:
        count += 1
        print('count = ', count)
        Text = wb_data.doc_text + '\n'
        if wb_data.weibo_source != "NULL":
            Text += wb_data.weibo_source
        #从微博text和原微博text中寻找关键词
        result_list = find_law(Text, sensitive_word_list, stopword_list,wb_data.weibo_id)
        
        #如果微博text或原微博text中存在关键词，就加入matched_law_data数据库
        #由于原数据库weibodata存在重复的微博数据，因此匹配后的微博数据也存在重复的，
        #即可能出现一条微博匹配上多部法律
        if len(result_list) >0:
            # print('result_list = ', result_list)
            for item in result_list:
                # print('keyword_name=',item[3])
                judge_law_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id, 
                    belong_to_law = item[2]
                )


#测试匹配法条级别的准确度
def clause_test_match():
    judge_clause_data.objects.all().delete()
    stopword_list = stopword.objects.all()
    sensitive_word_list = load_sensitive_word()
    wb_data_list = random_selected_data.objects.all()
    print('load test data completed!')
    count = 0
    flag=0
    for wb_data in wb_data_list:
        # if wb_data.weibo_id==4463260328107795:
        #     flag=1
        # else:
        #     flag=0
        # if flag==0:
        #     continue
        count += 1
        print('count = ', count)
        Text = wb_data.doc_text + '\n'
        if wb_data.weibo_source != "NULL":
            Text += wb_data.weibo_source
        # print('text = ',Text)
        result_list = match(Text, sensitive_word_list, stopword_list ,wb_data.weibo_id)
        
        if len(result_list) >0:
            # print('result_list = ', result_list)
            for item in result_list:
                if item[0] == 'law':
                    ref = multi_version_law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    # ref = law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    keyword_name = ref[0].law_name
                    element_name = ref[0].clause_name
                else:
                    ref = explain_element.objects.filter(explain_id = item[1], element_id = item[2])
                    keyword_name = ref[0].explain_name
                    element_name = ref[0].element_name
                # print('keyword_name = ', keyword_name, 'element_name = ', element_name)
            # print('Text = ', Text)
                judge_clause_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id,
                    belong_to_law = keyword_name, 
                    belong_to_clause =element_name,
                )


def auto_mark_nn_data():
    # nn_auto_label_data.objects.all().delete()
    stopword_list = stopword.objects.all()
    sensitive_word_list = load_sensitive_word()
    wb_data_list = nn_random_data.objects.filter(id__gt=3000)
    print('load test data completed!')
    count = 0
    flag=0
    for wb_data in wb_data_list:
        count += 1
        # if count <= 100:
        #     continue
        print('count = ', count)
        Text = wb_data.doc_text + '\n'
        if wb_data.weibo_source != "NULL":
            Text += wb_data.weibo_source
        # print('text = ',Text)
        result_list = match(Text, sensitive_word_list, stopword_list ,wb_data.weibo_id)
        
        if len(result_list) >0:
            # print('result_list = ', result_list)
            for item in result_list:
                if item[0] == 'law':
                    ref = multi_version_law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    # ref = law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    keyword_name = ref[0].law_name
                    element_name = ref[0].clause_name
                    content = ref[0].content
                else:
                    ref = explain_element.objects.filter(explain_id = item[1], element_id = item[2])
                    keyword_name = ref[0].explain_name
                    element_name = ref[0].element_name
                    content = ref[0].content

                nn_auto_label_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id,
                    belong_to_law = keyword_name, 
                    belong_to_clause =element_name,
                    law_id = item[1],
                    clause_id = item[2],
                    content = content,
                    weibo_content = Text,
                    label = 1
                )



def calculate_F1():
    #计算法律级别的F1
    real_result_list = real_law_data.objects.all()
    judge_result_list = judge_law_data.objects.all()
    #判断的所有类别
    # judge_classes = []
    #标注的所有类别
    # real_classes = []
    #所有标注数据的个数
    # total_exam_num = len(real_result_list)
    #各个类别的tf/fp/fn初始化为0
    total_law_TP = 0
    total_law_FP = 0
    total_law_FN = 0
    #把所有标注的类别加到列表中
    # for item in real_result_list:
    #     if item.belong_to_law not in real_classes:
    #         real_classes.append(item.belong_to_law)
    #把所有判断的类别加到列表中
    # for item in judge_result_list:
    #     if item.belong_to_law not in judge_classes:
    #         judge_classes.append(item.belong_to_law)

    #计算各类别总体TP
    #所有预测正确的个数
    #遍历标注数据列表
    for item in real_result_list:
        temp_list = judge_law_data.objects.filter(
            weibo_id=item.weibo_id,
            belong_to_law=item.belong_to_law
        )
        if len(temp_list)>0:
            total_law_TP += 1
    #计算各类别总体FP
    #就是算法给出的预测结果在实际标注数据中并不存在的个数
    #遍历判断数据列表，可能比标注数据列表短
    for item in judge_result_list:
        temp_list = real_law_data.objects.filter(
            weibo_id=item.weibo_id,
            belong_to_law=item.belong_to_law
        )
        if len(temp_list)==0:
            print('law_FP_example(weibo_id=',item.weibo_id,'belong_to_law=',item.belong_to_law,')')
            total_law_FP += 1
    #计算各类别总体FN
    #就是真实标注数据中没有被算法预测出来或者预测正确的个数
    #遍历标注数据列表
    for item in real_result_list:
        temp_list = judge_law_data.objects.filter(
            weibo_id=item.weibo_id,
            belong_to_law=item.belong_to_law
        )
        if len(temp_list)==0:
            print('law_FN_example(weibo_id=',item.weibo_id,'belong_to_law=',item.belong_to_law,')')
            total_law_FN += 1
    #计算精确率P
    P_law = total_law_TP / (total_law_TP + total_law_FP)
    #计算召回率R
    R_law = total_law_TP / (total_law_TP + total_law_FN)
    #计算F1
    F1_law = 2*total_law_TP / (2*total_law_TP + total_law_FP + total_law_FN)

    print("total_law_TP = ",total_law_TP)
    print("total_law_FP = ",total_law_FP)
    print("total_law_FN = ",total_law_FN)
    print("P_law = ",P_law)
    print("R_law = ",R_law)
    print("F1_law = ",F1_law)

    #计算法条级别的F1
    total_clause_TP = 0
    total_clause_FP = 0
    total_clause_FN = 0
    real_result_list = real_clause_data.objects.all()
    judge_result_list = judge_clause_data.objects.all()
    for item in real_result_list:
        temp_list = judge_clause_data.objects.filter(
            weibo_id=item.weibo_id,
            belong_to_law=item.belong_to_law,
            belong_to_clause=item.belong_to_clause
        )
        if len(temp_list)>0:
            total_clause_TP += 1
    for item in judge_result_list:
        temp_list = real_clause_data.objects.filter(
            weibo_id=item.weibo_id,
            belong_to_law=item.belong_to_law,
            belong_to_clause=item.belong_to_clause
        )
        if len(temp_list)==0:
            print('clause_FP_example(weibo_id=',item.weibo_id,'belong_to_law=',item.belong_to_law,'belong_to_clause=',item.belong_to_clause,')')
            total_clause_FP += 1
    for item in real_result_list:
        temp_list = judge_clause_data.objects.filter(
            weibo_id=item.weibo_id,
            belong_to_law=item.belong_to_law,
            belong_to_clause=item.belong_to_clause
        )
        if len(temp_list)==0:
            print('clause_FN_example(weibo_id=',item.weibo_id,'belong_to_law=',item.belong_to_law,'belong_to_clause=',item.belong_to_clause,')')
            total_clause_FN += 1
    #计算精确率P
    P_clause = total_clause_TP / (total_clause_TP + total_clause_FP)
    #计算召回率R
    R_clause = total_clause_TP / (total_clause_TP + total_clause_FN)
    #计算F1
    F1_clause = 2*total_clause_TP / (2*total_clause_TP + total_clause_FP + total_clause_FN)

    print("total_clause_TP = ",total_clause_TP)
    print("total_clause_FP = ",total_clause_FP)
    print("total_clause_FN = ",total_clause_FN)
    print("P_clause = ",P_clause)
    print("R_clause = ",R_clause)
    print("F1_clause = ",F1_clause)


def test_match():
    text = '''中华人民共和国水法》（第一章第三条、第十\n七条、
            第十九条、\n\r\n第三十条、第三十七条、第四十条）'''
    # obj = re.compile('第(.*?)条')
    obj = re.compile( r'(?s)第(.*?)条')
    find_list = obj.findall(text)
    # find_list = re.findall(text)
    print('find_list = ', find_list)


#不用了
def test_jieba():
    stopwords=load_stopwords()
    clauses = law_clause.objects.filter(law_name = "中华人民共和国公务员法")
    idf_dict = {}
    clause_num = len(clauses)
    text_list = []
    result_list = []
    for clause in clauses:
        t = clause.content.strip()
        result=t.find(clause.clause_name)
        if result != -1:
            while result != -1:
                tt=" "*len(clause.clause_name)
                t=t.replace(clause.clause_name,tt,1)
                result=t.find(clause.clause_name)
        text_list.append(t)
        # print(clause.content)
    # print(text_list)
    for text in text_list:
        seg_list = jieba.lcut(text)
        temp_list = []
        for word in seg_list:
            if word not in stopwords:
                if word not in ['\n','\t',' ','\u3000']:
                    if word not in temp_list:
                        temp_list.append(word)
                        if word not in idf_dict:
                            idf_dict[word]=1
                        else:
                            idf_dict[word]=idf_dict[word]+1
    
    for k,v in idf_dict.items():
        idf_dict[k]=math.log(clause_num/float(idf_dict[k]),10)
    L = sorted(idf_dict.items(),key=lambda item:item[1],reverse=True)
        
    # print("L = ",L)
    count = 0
    for text in text_list:
        count+=1
        print('count = ',count)
        seg_list = jieba.lcut(text)
        temp_list = []
        tf_dict = {}
        total_len = 0
        for word in seg_list:
            if word not in stopwords:
                if word not in ['\n','\t',' ','\u3000']:
                    if word not in temp_list:
                        temp_list.append(word)
                        if word not in tf_dict:
                            tf_dict[word]=1
                            total_len+=1
                        else:
                            tf_dict[word]=tf_dict[word]+1
                            total_len+=1
        for k,v in tf_dict.items():
            tf_dict[k]=(tf_dict[k]/float(total_len))*idf_dict[k]
        text_result_list = sorted(tf_dict.items(),key=lambda item:item[1],reverse=True)
        # print(text_result_list[:5])
        r = []
        index = 0
        for k in tf_dict:
            index+=1
            if index<=10:
                r.append(k)
        print("/".join(e for e in r))

def add_new_feature_words():
    stopwords=load_stopwords()
    # print(stopwords)
    multi_clause_list = multi_version_law_clause.objects.all()
    # multi_clause_list = multi_version_law_clause.objects.filter(law_id=36,clause_id =19)
    r1 = u'[a-zA-Z0-9’!"#$%&\'()\\*+,-./:;<=>?\u3000\n\ue010（） @，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    for mclause in multi_clause_list:
        text = re.sub(r1,'',mclause.content)
        # print('text=',text)
        seg_list = jieba.lcut(text)
        
        # seg_list = list(set(seg_list).difference(set(stopwords)))
        for index in range(len(seg_list)-1,-1,-1):
            if seg_list[index] in stopwords:
                # print('remove e = ',seg_list[index])
                seg_list.pop(index)
        
        # print('seg_list = ',seg_list)
        feature_list=('/'.join(e for e in seg_list))
        mclause.feature_words=feature_list
        mclause.save()


def add_feature_words():
    stopwords=load_stopwords()
    multi_law_list = multi_version_law.objects.all()
    for ml in multi_law_list:
        clauses = multi_version_law_clause.objects.filter(ful_name=ml.ful_name)
        idf_dict = {}
        clause_num = len(clauses)
        text_list = []
        result_list = []
        for clause in clauses:
            t = clause.content.strip()
            result=t.find(clause.clause_name)
            if result != -1:
                tt=" "*len(clause.clause_name)
                t=t.replace(clause.clause_name,tt)
            text_list.append(t)

        for text in text_list:
            seg_list = jieba.lcut(text)
            temp_list = []
            for word in seg_list:
                if word not in stopwords:
                    if word not in ['\n','\t',' ','\u3000']:
                        if word not in temp_list:
                            temp_list.append(word)
                            if word not in idf_dict:
                                idf_dict[word]=1
                            else:
                                idf_dict[word]=idf_dict[word]+1
    
        for k,v in idf_dict.items():
            idf_dict[k]=math.log(clause_num/float(idf_dict[k]),10)
        L = sorted(idf_dict.items(),key=lambda item:item[1],reverse=True)

        count = 0
        for text in text_list:
            count+=1
            print('count = ',count)
            seg_list = jieba.lcut(text)
            temp_list = []
            tf_dict = {}
            total_len = 0
            for word in seg_list:
                if word not in stopwords:
                    if word not in ['\n','\t',' ','\u3000']:
                        if word not in temp_list:
                            temp_list.append(word)
                            if word not in tf_dict:
                                tf_dict[word]=1
                                total_len+=1
                            else:
                                tf_dict[word]=tf_dict[word]+1
                                total_len+=1
            for k,v in tf_dict.items():
                tf_dict[k]=(tf_dict[k]/float(total_len))*idf_dict[k]
            text_result_list = sorted(tf_dict.items(),key=lambda item:item[1],reverse=True)
            print('origin_list = ',text_result_list)
            print('top5 = ',text_result_list[:5])
            
            r = []
            index = 0
            # for k in tf_dict:
            #     index+=1
            #     score=str('%.2f' % tf_dict[k])
            #     if index<=10:
            #         r.append(k+'='+score)
            for u in text_result_list:
                index+=1
                score=str('%.2f' % u[1])
                if index<=10:
                    r.append(u[0]+'='+score)
            feature_words=("/".join(e for e in r))
            print('feature_words = ',feature_words)
            target = multi_version_law_clause.objects.filter(ful_name=ml.ful_name,clause_id = count)
            if len(target)>0:
                target[0].spacial_words = feature_words
                target[0].save()
            


def test_trim_shuming():
    text = '// 【[话筒]中办印发《意见》，要求#为基层公务员松绑减负#】近日，中共中央办公厅印发了《关于贯彻实施公务员法建设高素质专业化公务员队伍的意见》。《意见》要求着力纠正形式主义、官僚主义，严肃査处不担当、不作为、乱作为等问题。《关于贯彻实施公务员法建设高素质专业化公务员队伍的意见》突出重视基层导向，切实为基层公务员松绑减负。http://t.cn/AijuGdIL ??'
    obj = re.compile('《(.+?)》')
    result_list = obj.findall(text)
    print('result_list =',result_list)
    keyword_list = ['公务员法','宪法','刑法']
    remove_list=[]
    for shuming in result_list:
        for keyword in keyword_list:
            if keyword in shuming and len(shuming)-len(keyword)>3:
                print('keyword = ', keyword,'shuming=',shuming)
                remove_list.append(shuming)
        if len(remove_list)>0:
            for remove_word in remove_list:
                result = text.find(remove_word)
                if result != -1:
                    replace_word = " "*len(remove_word)
                    text = text.replace(remove_word,replace_word)
    print('text = ',text)



def test_section():
    l1=[1,2,3]
    l2=[]
    joint_list = list(set(l1).intersection(set(l2)))
    print('join_list=',joint_list)



