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

def RemoveTrashWord(text):
    l=['doge','二哈','ok','jb','嗝','噫','嚯','費解','疑问','笑cry','偷笑','你妈','艹','丫','n','吐','扯淡','泪','泪目','···','good','渣渣','怒','吐血','哇','啊','呀','呢','吃瓜','微笑','嘻嘻','哦','欸','拜拜','允悲','费解','棒棒哦','等一蛤','等一哈','赞','鲜花','心','抱歉，由于作者设置，你暂时没有这条微博的查看权限。查看帮助','哈','有关规定']
    for word in l:
        result=text.find(word)
        if result !=-1:
            text=text.replace(word,'')
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


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
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
            result_t=()
            if t[1]=='explain':
                # result_t = ('explain', t[2],element_name)
                temp_l = explain_element.objects.filter(explain_name=t[2],element_name = element_name)
                if len(temp_l)>0:
                    result_t=('explain',temp_l[0].explain_id,temp_l[0].element_id)
            else:
                # result_t = ('law', t[2],element_name)
                temp_l = law_clause.objects.filter(law_name = t[2],clause_name = element_name)
                if len(temp_l)>0:
                    result_t = ('law',temp_l[0].law_id,temp_l[0].clause_id)

            # result_t = match_multi_version_clause(text,t[1],t[2],element_name)
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
    
    return result_list



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
        text = wb_data.doc_text
        if wb_data.weibo_source != "NULL":
            text += wb_data.weibo_source

        text=CleanText(text)
        text=RemoveTrashWord(text)
        result_list = match(text, sensitive_word_list, stopword_list ,wb_data.weibo_id)
        
        if len(result_list) >0:
            # print('result_list = ', result_list)
            for item in result_list:
                if item[0] == 'law':
                    # ref = multi_version_law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    ref = law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                    keyword_name = ref[0].law_name
                    element_name = ref[0].clause_name
                else:
                    ref = explain_element.objects.filter(explain_id = item[1], element_id = item[2])
                    keyword_name = ref[0].explain_name
                    element_name = ref[0].element_name
                # keyword_name=item[1]
                # element_name=item[2]
                judge_clause_data.objects.get_or_create(
                    weibo_id = wb_data.weibo_id,
                    belong_to_law = keyword_name, 
                    belong_to_clause =element_name,
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



def buchong():
    # l = [4353765194335517,]
    l = nn_fine_grain_training_data.objects.all()
    L = len(l)
    print('L = ',L)
    index_list = []
    count = 0
    while count<200:
        r = random.randint(0,L-1)
        if r not in index_list:
            index_list.append(r)
            count+=1
    index_list = sorted(index_list)
    weibo_id_list = []
    positive_flag = 0
    for index in index_list:
        temp_l = l[index:index+1]
        t=temp_l[0]
        label = t.label
        weibo_id =t.weibo_id
        if weibo_id not in weibo_id_list:
            weibo_id_list.append(weibo_id)
        belong_to_law =t.belong_to_law
        belong_to_clause = t.belong_to_clause
        if label == 1:
            positive_flag += 1
            pass
            real_clause_data.objects.get_or_create(
                weibo_id = weibo_id,
                belong_to_law=belong_to_law,
                belong_to_clause=belong_to_clause
            )
    print('positive_flag = ',positive_flag)
    print('len of weibo_id  =',len(weibo_id_list))
    for weiboid in weibo_id_list:
        data_list = solr_weibo_data.objects.filter(weibo_id = weiboid)
        if len(data_list)>0:
            t = data_list[0]
            random_selected_data.objects.get_or_create(
                weibo_link = t.weibo_link,
                weibo_id = t.weibo_id,
                tou_xiang = t.tou_xiang,
                author_name = t.author_name,
                doc_text = t.doc_text,
                weibo_source = t.weibo_source,
                doc_date = t.doc_date
            )

def kuochong():
    l = random_selected_data.objects.filter(weibo_id = 4461113464631557)
    if len(l)>0:
        t = l[0]
        for i in range(15):
            new_id = i + 4461113464631557
            random_selected_data.objects.get_or_create(
                weibo_id = new_id,
                weibo_link = t.weibo_link,
                tou_xiang = t.tou_xiang,
                author_name = t.author_name,
                doc_text = t.doc_text,
                weibo_source = t.weibo_source,
                doc_date = t.doc_date
            )
    temp_list = real_clause_data.objects.filter(weibo_id = 4461113464631557)
    if len(temp_list)>0:
        t = temp_list[0]
        for i in range(15):
            new_id = i + 4461113464631557
            real_clause_data.objects.get_or_create(
                weibo_id = new_id,
                belong_to_law = t.belong_to_law,
                belong_to_clause = t.belong_to_clause
            )