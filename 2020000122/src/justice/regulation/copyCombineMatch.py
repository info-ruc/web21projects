#!/usr/bin/env python
#coding:utf-8
import copy
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

def cal_sentence_dtw_score(c_list,s_list,frame_width):
    score=0
    # print('frame_width = ',frame_width)
    m = len(c_list)
    n = len(s_list)
    print('m = ',m,' n = ',n)
    print('c_list = ',c_list)
    print('s_list = ',s_list)
    match_array = np.zeros((m,n),dtype=int)
    for i in range(m):
        for j in range(n):
            match_array[i][j] = 1 if c_list[i]==s_list[j] else 0
            # print(c_list[i])
    #print(match_array)
    if frame_width == 16:
        k1 = np.eye(16,k=1)
        k1 = 0.5*k1
        k2 = np.eye(16,k=-1)
        k2 = 0.5*k2
        k3 = np.identity(16)
        kernel = k1+k2+k3
    elif frame_width == 8:
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

def veto(text):
    veto_words=['吴青峰','环球音乐','EXO','韩国现行兵役法','韩国兵役法','防弹少年','奥地利','法国大革命','日本男子起诉国家','韩国瑜','人权基本法','美国买个书号','那就是美利坚合众国','俄罗斯总统普京','梅德韦杰夫']
    for word in veto_words:
        result=text.find(word)
        if result != -1:
            return 1
    return 0

def TotalClean(text):
    R = u'[’//#$%&→↓￥〔〕,.，。；;、~《》<>\：\'()——*+-/<=>～\r \r\n\u3000\t\n\ue010（） @★…【】“”‘’！？?![\\]^_`\\{|}~\\\\]+'
    text=re.sub(R,'',text)
    return text


def CleanText(text):
    huanhang=u'[\r\n\t]+'
    text=re.sub(huanhang,'',text)
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
    # zhanghao = re.compile('@.+? ')
    # result_list=zhanghao.findall(text)
    # for t in result_list:
    #     print('t = ',t)
    #     text=text.replace(t,"")
    zhuanfa=re.compile('转发微博')
    result_list=zhuanfa.findall(text)
    for t in result_list:
        print('t = ',t)
        text=text.replace(t,"")
    R = u'[’//#$%&→↓￥〔〕：\'()——*+-/<=>～\r \r\n\u3000\t\n\ue010（） @★…【】“”‘’！？?![\\]^_`\\{|}~\\\\]+'
    text=re.sub(R,'',text)
    return text

def RemovePuc(text):
    R = u'[’//\\\\#$%&〔〕：：、；。，,.;?\'()\\*+-/<=>\r \r\n\u3000\t\n\ue010（）　 （） @★…【】“”‘’！？?![\\]^_`{|}~]+'
    text=re.sub(R,'',text)
    return text

def WashText(text):
    r1=u'[\ue010\u3000\r\n\r\t《》<> ]+'
    text=re.sub(r1,'',text)
    juhao=re.compile(u'(。[。 ,]+)')
    result_list = juhao.findall(text)
    for t in result_list:
        text=text.replace(t,"")
    head=re.compile('( 。)')
    result_list = head.findall(text)
    for t in result_list:
        text=text.replace(t,"")
    return text

def RemoveTrashWord(text):
    l=['doge','二哈','ok','jb','嗝','噫','嚯','費解','疑问','笑cry','偷笑','你妈','艹','丫','n','吐','扯淡','泪','泪目','···','good','渣渣','怒','吐血','哇','啊','呀','呢','吃瓜','微笑','嘻嘻','哦','欸','拜拜','允悲','费解','棒棒哦','等一蛤','等一哈','赞','鲜花','心','抱歉，由于作者设置，你暂时没有这条微博的查看权限。查看帮助','哈','有关规定']
    for word in l:
        result=text.find(word)
        if result !=-1:
            text=text.replace(word,'')
    return text


def JudgeNoise(text):
    if len(text)<=12:
        print('Noise Text = ',text)
        return 1
    return 0

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

def convert2Chinese(num):
    _MAPPING = (u'零', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十', u'十一', u'十二', u'十三', u'十四', u'十五', u'十六', u'十七',u'十八', u'十九')
    _P0 = (u'', u'十', u'百', u'千',)
    _S4 = 10 ** 4
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

def pre_match(text,match_word_list,stopword_list):
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
    return temp_list,text

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



class Option:
    def __init__(self, sequence_score, tf_idf_score, common_part_score, effectiveTime,law_id,clause_id,law_name,clause_name,clause_content):
        self.sequence_score = sequence_score
        self.tf_idf_score = tf_idf_score
        self.common_part_score = common_part_score
        self.effectiveTime = effectiveTime
        self.law_id=law_id
        self.clause_id=clause_id
        self.law_name=law_name
        self.clause_name=clause_name
        self.clause_content=clause_content


def RemoveDuplicate(score_list):
    clause_content_list = []
    remove_index_list=[]
    score_list=sorted(score_list,key=lambda x:-x.effectiveTime)
    for i in range(len(score_list)):
        content=RemovePuc(score_list[i].clause_content)
        if content not in clause_content_list:
            clause_content_list.append(content)
            # print('append = ',content)
        else:
            # print('remove = ',content)
            remove_index_list.append(i)
    for index in range(len(remove_index_list)-1,-1,-1):
        # print('Remove = ',score_list[index].clause_content)
        score_list.pop(index)
    return score_list


#t=（position位置,'law',中华人民共和国公务员法）
#（position位置,'explain',...）
def SelectOption(text,t,weibo_id):
    stopwords = load_stopwords()
    result_list=[]
    clause_list = multi_version_law_clause.objects.filter(law_name=t[2])
    eftime_list=[]
    m_law_list = multi_version_law.objects.filter(law_name = t[2])
    for mlaw in m_law_list:
        eftime_list.append((mlaw.ful_name,mlaw.effectiveDate))

    if len(clause_list)>0:
        for word in stopwords:
            if word in text:
                text=text.replace(word,'')
        seg_list = jieba.lcut(text)
        score_list=[]
        if len(seg_list)>0:
            for cl in clause_list:
                effectiveTime=0
                for eftime in eftime_list:
                    if cl.ful_name==eftime[0]:
                        effectiveTime = eftime[1]
                        break

                feature_list = cl.feature_words.split('/')
                common_part = set(feature_list).intersection(set(seg_list))
                lc = len(common_part)
                lm = min(len(feature_list),len(seg_list))
                common_part_score = lc/float(lm)

                
                if lc>3:
                    if lc>=8:
                        frame_width = 16
                        sequence_score = 2*cal_sentence_dtw_score(feature_list,seg_list,frame_width)
                    else:
                        frame_width = 8
                        sequence_score = cal_sentence_dtw_score(feature_list,seg_list,frame_width)
                else:
                    sequence_score=0
                
                tf_score=0.0
                total_tf_score=0.0
                spacial_words=cl.spacial_words.split('/')
                for w in spacial_words:
                    try:
                        spacial_word,weight=w.split('=')
                        weight=float(weight)
                        if spacial_word in seg_list:
                            tf_score+=weight
                        total_tf_score+=weight
                    except:
                        continue
                if total_tf_score==0.0:
                    print('999')
                    tf_idf_score=0.0
                else:
                    tf_idf_score=float(tf_score)/float(total_tf_score)
                print('clause_id = ',cl.clause_id)
                print('tf_score = ',tf_score,'tf-idf_score = ',tf_idf_score,'total_score = ',total_tf_score)
                print('lc = ',lc,'sequ_score=',sequence_score,'tf-score=',tf_idf_score,'com_score=',common_part_score)
                if sequence_score>0.4 or common_part_score>0.3 or tf_idf_score>0.4:
                    score_list.append(Option(sequence_score,tf_idf_score,common_part_score,effectiveTime,cl.law_id,cl.clause_id,cl.law_name,cl.clause_name,cl.content))

            #返回值result_list = [(label,weibo_id,weibo_text,clause_content,law_name,clause_name,law_id,clause_id),()]
            if len(score_list) > 0:
                score_list = RemoveDuplicate(score_list)
                score_list.sort(key=lambda x:(-x.sequence_score,-x.tf_idf_score,-x.common_part_score,-x.effectiveTime))
                print('↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')
                print('len(score_list)=',len(score_list))
                # print('score_list = ',score_list)
                for u in score_list:
                    print('clause_id = ',u.clause_id)
                    print('(%.3f,%.3f,%.3f)'%(u.sequence_score,u.tf_idf_score,u.common_part_score))
                option_num=0
                for item in score_list:
                    if option_num==3:
                        break
                    if option_num==0:
                        if item.sequence_score>=2.0:
                            # label='%.2f' % max(item.sequence_score,item.tf_idf_score,item.common_part_score)
                            result_list.append((t[1],item.law_id,item.clause_id))
                            break
                        elif item.sequence_score>=0.8 or item.tf_idf_score>=0.6 or item.common_part_score>=0.4:
                            result_list.append((t[1],item.law_id,item.clause_id))
                            break
                            # label='%.2f' % max(item.sequence_score,item.tf_idf_score,item.common_part_score)
                        else:
                            # label=0
                            # result_list.append((t[1],item.law_id,item.clause_id))
                            label='%.2f' % max(item.sequence_score,item.tf_idf_score,item.common_part_score)
                        temp_turple=(label,weibo_id,text,item.clause_content,item.law_name,item.clause_name,item.law_id,item.clause_id)
                        result_list.append(temp_turple)
                    else:
                        # label=0
                        label='%.2f' % max(item.sequence_score,item.tf_idf_score,item.common_part_score)
                        temp_turple=(label,weibo_id,text,item.clause_content,item.law_name,item.clause_name,item.law_id,item.clause_id)
                        result_list.append(temp_turple)
                    option_num+=1

    print('text = ',text)
    print('result_list = ',result_list)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return result_list


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


#返回值result_list = [(label,weibo_id,weibo_text,clause_content,law_name,clause_name,law_id,clause_id),()]
def match_item_in_one_area(text, t, weibo_id):
    print('weibo_id = ',weibo_id)
    print('text = ', text)
    if t[1]=='others':
        return []
    text=WashText(text)
    flag=JudgeNoise(text)
    if flag==1:
        return []
    result_list = []
    obj = re.compile('第(.{1,5})条')
    find_list = obj.findall(text)
    #如果在text中有出现第几条
    if len(find_list) >0:
        
        old_list = find_list
        find_list = recheck(find_list)
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
        return result_list
    #如果没有明确出现第几条，就根据语义相似度匹配，这个过程只针对law，对于司法解释没有处理
    else:
        return SelectOption(text,t,weibo_id)


#法条级别匹配核心算法，传入文本、匹配词列表和微博ID，返回匹配结果列表result_list
#返回值result_list = (‘law’, 23(法律ID), 45(法条ID))
#返回值result_list = (‘explain’, 23(解释ID), 45(解释条ID))
#返回值result_list = [(label,weibo_id,weibo_text,clause_content,law_name,clause_name,law_id,clause_id),()]
def match(text, match_word_list, stopword_list,weibo_id):
    temp_list,text=pre_match(text, match_word_list, stopword_list)
    print('len ==== : ',len(temp_list))
    if len(temp_list) == 0:
        return []
    elif len(temp_list) == 1:
        #如果只有一部法律就全文匹配
        result_list = match_item_in_one_area(text,temp_list[0],weibo_id)
        return result_list
    else:
        #按照出现的位置从前往后排列
        old_list = sorted(temp_list, key = lambda x:x[0])
        temp_list = trim_list(old_list)
        result_list = []
        for index in range(len(temp_list)):
            begin = temp_list[index][0]
            if index == 0:
                begin = 0
            end = -1 if index == (len(temp_list)-1) else temp_list[index+1][0]
            l = match_item_in_one_area(text[begin:end], temp_list[index],weibo_id)
            for item in l:
                result_list.append(item)
        return result_list

def get_fine_training_data():
    #加载停用词和关键词
    stopword_list = stopword.objects.all()
    sensitive_word_list = load_sensitive_word()
    #清除原来的数据
    nn_fine_grain_training_data.objects.all().delete()
    #取数据
    random_data_list = nn_random_data.objects.all()
    random_data_list = sorted(random_data_list,key=lambda x:x.doc_date,reverse=False)
    #遍历循环数据列表
    data_count=0
    for data in random_data_list:
        data_count+=1
        if data_count % 100 ==0:
            print(data_count)
        text=data.doc_text
        if data.weibo_source != 'NULL':
            text += data.weibo_source
        flag = veto(text)
        if flag == 1:
            continue
        text=CleanText(text)
        text=RemoveTrashWord(text)
        #返回值result_list = [(label,weibo_id,weibo_text,clause_content,law_name,clause_name,law_id,clause_id),()]
        result_list = match(text, sensitive_word_list, stopword_list, data.weibo_id)
        
        if len(result_list)>0:
            for item in result_list:
                nn_fine_grain_training_data.objects.get_or_create(
                    label=item[0],
                    weibo_id=item[1],
                    weibo_content=item[2],
                    content=item[3],
                    belong_to_law=item[4],
                    belong_to_clause=item[5],
                    law_id=item[6],
                    clause_id=item[7]
                )
        
    print('get_train_data done!')


#测试匹配法条级别的准确度
def clause_test_match():
    output1 = open('300_test_file.csv','wb')
    output2 = open('temp_result.txt','wb')
    output1.write('sentence1\tsentence2\tLabel\n'.encode('utf-8'))
    # output2.write('sentence1\tsentence2\tLabel\n'.encode('utf-8'))
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
        print('text = ',text)
        veto_flag = veto(text)
        if veto_flag == 1:
            print('veto = ',wb_data.weibo_id)
            continue
        text=CleanText(text)
        text=RemoveTrashWord(text)
        print('weibo_id = ',wb_data.weibo_id)
        print('text = ',text)
        result_list = match(text, sensitive_word_list, stopword_list ,wb_data.weibo_id)
        
        if len(result_list) >0:
            for item in result_list:
                if len(item) == 3:
                    if item[0] == 'law':
                        ref = multi_version_law_clause.objects.filter(law_id = item[1], clause_id = item[2])
                        keyword_name = ref[0].law_name
                        element_name = ref[0].clause_name
                    else:
                        ref = explain_element.objects.filter(explain_id = item[1], element_id = item[2])
                        keyword_name = ref[0].explain_name
                        element_name = ref[0].element_name

                    judge_clause_data.objects.get_or_create(
                        weibo_id = wb_data.weibo_id,
                        belong_to_law = keyword_name, 
                        belong_to_clause =element_name,
                    )
                else:
                    clause_content = CleanText(item[3])
                    clause_content = WashText(clause_content)
                    clause_content = RemoveTrashWord(clause_content)
                    
                    output1.write(CleanText(item[2]).encode('utf-8'))
                    output1.write('\t'.encode('utf-8'))
                    output1.write(clause_content.encode('utf-8'))
                    output1.write('\t'.encode('utf-8'))
                    output1.write(str('%0.1f' % float(item[0])).encode('utf-8'))
                    output1.write('\n'.encode('utf-8'))

                    output2.write(str(item[1]).encode('utf-8'))
                    output2.write('\t'.encode('utf-8'))
                    output2.write(item[4].encode('utf-8'))
                    output2.write('\t'.encode('utf-8'))
                    output2.write(item[5].encode('utf-8'))
                    output2.write('\n'.encode('utf-8'))

    output1.close()
    output2.close()




def gen_train_data_file():
    fout=open('3500_fine_train_data.csv','wb')
    fout.write('sentence1\tsentence2\tLabel\n'.encode('utf-8'))
    l = nn_fine_grain_training_data.objects.all()
    # l=l[:1270]
    text_list = []
    id_list = []
    for data in l:
        label = data.label
        if label==1:
            Label='1.0'
        else:
            Label='0.0'
        myid=data.weibo_id
        # temp_list = solr_weibo_data.objects.filter(weibo_id=myid)
        # if len(temp_list)>0:
        #     weibo_content = '说一个有趣的现象，或者说悖论吧。\n\n反垄断大棒挥舞的最积极的欧盟，没出现一个牛逼的互联网公司；而对反垄断保持谦抑的美国和中国，恰恰是市场竞争最为激烈，大公司层出不穷。\n\n美国从微软开始，雅虎、谷歌、亚马逊、脸书、奈飞……\n中国则有BAT、头条、京东、拼多多…'+temp_list[0].doc_text
        #     if temp_list[0].weibo_source!='NULL':
        #         weibo_content+=temp_list[0].weibo_source
        # else:
        #     print('baga')
        #     continue
        weibo_content = data.weibo_content
        text=CleanText(weibo_content)
        text=WashText(text)
        text=RemoveTrashWord(text)
        text=TotalClean(text)
        clause_content = data.content
        clause_content = CleanText(clause_content)
        clause_content = WashText(clause_content)
        clause_content = RemoveTrashWord(clause_content)
        clause_content = TotalClean(clause_content)

        if text not in text_list:
            text_list.append(text)
            id_list.append(myid)
        else:
            if myid not in id_list:
                continue
        fout.write(text.encode('utf-8'))
        fout.write('\t'.encode('utf-8'))
        fout.write(clause_content.encode('utf-8'))
        fout.write('\t'.encode('utf-8'))
        fout.write(Label.encode('utf-8'))
        fout.write('\n'.encode('utf-8'))
    fout.close()





def shuffle():
    fin1=open('3500_fine_train_data.csv','r')
    # fin2=open('my_0405_fine_train_data.csv','r')
    fout=open('new_only_shuffle_data.csv','wb')
    l = []
    count=0
    for line in fin1:
        count+=1
        if count==1:
            continue
        if line not in l:
            l.append(line)
    # count=0
    # for line in fin2:
    #     count+=1
    #     if count==1:
    #         continue
    #     if line not in l:
    #         l.append(line)
    for i in range(30):
        random.shuffle(l)
    fout.write('sentence1\tsentence2\tLabel\n'.encode('utf-8'))
    for line in l:
        fout.write(line.encode('utf-8'))
    fout.close()

def split_train_test_set():
    fin = open('new_shuffle_data.csv','r')
    data_list = []
    count=0
    for line in fin:
        count+=1
        if count==1:
            continue
        data_list.append(line)
    L = len(data_list)
    print('L = ',L)
    for i in range(10):
        random.shuffle(data_list)
    
    l=[]

    for i in range(5):
        begin = int(0.2*i*L)
        end = int(0.2*(i+1)*L)
        print('begin = ',begin,' end = ',end)
        test_list = data_list[int(0.2*i*L):int(0.2*(i+1)*L)]
        l.append(test_list)
    
    print('len(l) = ',len(l))

    
    for i in range(5):
        print('i = ',i)
        total_list = copy.deepcopy(l)
        dir_name = 'mydata'+str(i+1)+'/'
        dir_path = os.path.join(os.curdir,dir_name)
        os.mkdir(dir_path)
        fout1=open(dir_path+'train.csv','wb')
        fout2=open(dir_path+'test.csv','wb')
        fout1.write('sentence1\tsentence2\tLabel\n'.encode('utf-8'))
        fout2.write('sentence1\tsentence2\tLabel\n'.encode('utf-8'))
        test_list = total_list.pop(i)
        print('len(test_list) = ',len(test_list))
        print('len(total_list) = ',len(total_list))
        train_list = []
        for temp_list in total_list:
            train_list = train_list + temp_list
        print('len(train_list) = ',len(train_list))
        
        
        for line in train_list:
            fout1.write(line.encode('utf-8'))
        for line in test_list:
            fout2.write(line.encode('utf-8'))
        fout1.close()
        fout2.close()
    




