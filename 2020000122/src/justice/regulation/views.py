from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
import datetime
import json
import re

from .models import alias
from .models import law
from .models import law_clause
from .models import multi_version_law
from .models import multi_version_law_clause
from .models import explain
from .models import explain_element
from .models import solr_weibo_data
from .models import matched_clause_data
from .models import matched_law_data
from .models import law_charts_data
from .models import explain_charts_data
from .models import random_selected_data
from .models import real_law_data
from .models import real_clause_data
from .models import judge_law_data
from .models import judge_clause_data
from .models import nn_random_data
from .models import nn_auto_label_data
from .models import nn_fine_grain_training_data

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

def delete_nn_data(request):
    if request.method=='POST':
        print('into the function')
        data_id=request.POST.get('data_id')
        print('data_id=',data_id)
        print(type(data_id))
        l = nn_fine_grain_training_data.objects.filter(id=data_id)
        print('data_id = ',data_id)
        if len(l)>0:
            l[0].label=2
            l[0].save()
            print('success')
            return render(request,'regulation/nn_fine_biaozhu.html')


def change_nn_data_label(request):
    if request.method=='POST':
        print('into the function')
        data_id=request.POST.get('data_id')
        print('data_id=',data_id)
        print(type(data_id))
        l = nn_auto_label_data.objects.filter(id=data_id)
        print('data_id = ',data_id)
        if len(l)>0:
            l[0].label=0
            l[0].save()
            print('success')
            return render(request,'regulation/nn_biaozhu.html')

def show_5000_random_data(request):
    l=nn_random_data.objects.all()
    context={'data_list':l}
    return render(request, 'regulation/biaozhu.html', context)

def show_new_training_data(request):
    l=nn_fine_grain_training_data.objects.all()
    context={'data_list':l}
    return render(request, 'regulation/show_new_training_data.html', context)

def positive_data(request):
    if request.method=='POST':
        data_id=request.POST.get('data_id')
        print('data_id=',data_id)
        print(type(data_id))
        l = nn_fine_grain_training_data.objects.filter(id=data_id)
        print('data_id = ',data_id)
        if len(l)>0:
            l[0].label=1
            l[0].save()
            print('success')
            return render(request,'regulation/show_new_training_data.html')

def negative_data(request):
    if request.method=='POST':
        data_id=request.POST.get('data_id')
        print('data_id=',data_id)
        print(type(data_id))
        l = nn_fine_grain_training_data.objects.filter(id=data_id)
        print('data_id = ',data_id)
        if len(l)>0:
            l[0].label=0
            l[0].save()
            print('success')
            return render(request,'regulation/show_new_training_data.html')


def show_random_data(request):
    l=random_selected_data.objects.all()
    context={'data_list': l}
    return render(request, 'regulation/biaozhu.html', context)

def show_fine_grain_training_data(request):
    l=nn_fine_grain_training_data.objects.all()
    context={'data_list': l}
    return render(request, 'regulation/nn_fine_biaozhu.html', context)

def show_nn_training_data(request):
    l=nn_auto_label_data.objects.all()
    context={'data_list': l}
    return render(request, 'regulation/nn_biaozhu.html', context)

def dynamic_display(request):
    if request.method=='POST':
        keyword_name=request.POST.get('keyword_name')
        keyword_id=request.POST.get('keyword_id')
        start_time=request.POST.get('starttime')
        end_time=request.POST.get('endtime')
        option = request.POST.get('option')

        print('keyword_name',keyword_name)
        print('keyword_id',keyword_id)
        print('starttime=',start_time)
        print('endtime=',end_time)
        print('option=',option)

        sy,sm,sd = start_time.split('/')
        ey,em,ed = end_time.split('/')
        start_date = datetime.date(int(sy), int(sm), int(sd))
        end_date = datetime.date(int(ey), int(em), int(ed))

        #从数据库中查询数据放到data_list中
        if option=='origin':
            data_list = matched_law_data.objects.filter(keyword_name= keyword_name,keyword_id = keyword_id, doc_date__range=(start_date, end_date),origin_tag=1)
        elif option=='source':
            data_list = matched_law_data.objects.filter(keyword_name= keyword_name,keyword_id = keyword_id, doc_date__range=(start_date, end_date),source_tag=1)
        print('len = ',len(data_list))
        if len(data_list)>1000:
            print('old len = ',len(data_list))
            data_list = data_list[0:1000]
            print('new len = ',len(data_list))
        # data_list = sorted(data_list, key=lambda x:x.doc_time, reverse=False)

        context = {
            'data_list':data_list,
        }
        if option=='origin':
            return render(request,'regulation/origin_part_weibo.html', context)
        elif option=='source':
            return render(request,'regulation/source_part_weibo.html', context)
        # return render_to_response('regulation/test.html', context)
        # html = render_to_response("regulation/test.html", {"context": context})
        # return HttpResponse(context)



# 视图函数
def law_list(request):
    laws=law.objects.all()
    context={'laws': laws}
    return render(request, 'regulation/law_list.html', context)


def law_clause_list(request, param):
    myid, time_range = param.split('@')
    print('myid =|'+myid+'|')
    ID = int(myid)

    start_time, end_time = time_range.split('&')
    sy,sm,sd = start_time.split('-')
    ey,em,ed = end_time.split('-')
    start_date = datetime.date(int(sy), int(sm), int(sd))
    end_date = datetime.date(int(ey), int(em), int(ed))

    # print("started!")
    # start_date = datetime.date(2019, 1, 1)
    # end_date = datetime.date(2019, 2, 3)

    # clauses = law_clause.objects.filter(law_id = ID)
    title = law.objects.get(law_id = ID).law_name
    origin_data_list = matched_law_data.objects.filter(data_type='law', keyword_id=ID,doc_date__range=(start_date,end_date),origin_tag=1)
    data_list = matched_law_data.objects.filter(data_type = 'law', keyword_id = ID,doc_date__range=(start_date, end_date),source_tag=1)
    # data_list = sorted(data_list, key=lambda x:x.doc_time, reverse=False)
    if len(origin_data_list)>1000:
        origin_data_list=origin_data_list[0:1000]
    
    if len(data_list)>1000:
        print('old len = ',len(data_list))
        data_list = data_list[0:1000]
        print('new len = ',len(data_list))

    
    #判断是否有历史版本，以及排除掉当前页显示的法律版本，显示历史版本
    h_list = []
    history_version_list = multi_version_law.objects.filter(law_name = title,timeliness="现行有效")
    if len(history_version_list)==0:
        h_list = multi_version_law.objects.filter(law_name = title).exclude(timeliness = "尚未生效")
        tl = multi_version_law.objects.filter(law_name = title, timeliness="尚未生效")
        if len(tl)>0:
            ful_name = tl[0].ful_name
            clauses = multi_version_law_clause.objects.filter(ful_name = ful_name)
    else:
        h_list = multi_version_law.objects.filter(law_name = title).exclude(timeliness = "现行有效")
        tl = multi_version_law.objects.filter(law_name=title,timeliness="现行有效")
        if len(tl)>0:
            ful_name=tl[0].ful_name
            clauses = multi_version_law_clause.objects.filter(ful_name = ful_name)


    print(len(h_list))
    context = {
        'id':ID,
        'title':title,
        'clauses': clauses,
        'origin_data_list':origin_data_list,
        'data_list':data_list,
        'history_version':h_list,
        'ful_name':ful_name,
    }
    #画历史图表需要的数据
    data={
        'date_list':[],
        'total_list':[],
        'origin_total_list':[],
        'rate_list':[],
        'alias':[],
    }
    l=law_charts_data.objects.filter(law_id=ID)
    for item in l:
        data['date_list'].append(item.date.strftime("%Y/%m/%d"))
        data['total_list'].append(item.total)
        data['origin_total_list'].append(item.origin_count)
        data['rate_list'].append(item.rate)
    
    alias_list = alias.objects.filter(data_type='law',keyword_id=myid)
    if len(alias_list)>0:
        for item in alias_list:
            data['alias'].append(item.alias_name)
    if len(h_list)>0:
        return render(request, 'regulation/law_clause_multi_version.html', {'context':context,'Data':json.dumps(data)})
    else:
        return render(request, 'regulation/law_clause_list.html', {'context': context,'Data': json.dumps(data)})


def history_version_display(request, id):
    print(id)
    clauses = multi_version_law_clause.objects.filter(law_id = id)
    l = multi_version_law.objects.filter(law_id=id)
    ful_name = ""
    if len(l)>0:
        ful_name = l[0].ful_name
    print('ful_name = ',ful_name)
    context = {
        'clauses': clauses,
        'ful_name':ful_name,
    }
    return render(request, 'regulation/history_version.html', context)



def explain_list(request):
    explains=explain.objects.all()
    context={'explains': explains}
    return render(request, 'regulation/explain_list.html', context)


def explain_element_list(request, param):
    print('param='+param)
    myid, time_range = param.split('@')
    print('myid =|'+myid+'|')
    ID = int(myid)
    # ID = myid

    start_time, end_time = time_range.split('&')
    sy,sm,sd = start_time.split('-')
    ey,em,ed = end_time.split('-')
    start_date = datetime.date(int(sy), int(sm), int(sd))
    end_date = datetime.date(int(ey), int(em), int(ed))

    # print("started")
    elements = explain_element.objects.filter(explain_id = ID)
    title = explain.objects.get(explain_id = ID).explain_name
    origin_data_list=matched_law_data.objects.filter(data_type = 'explain', keyword_id = ID, doc_date__range=(start_date, end_date),origin_tag=1)
    data_list = matched_law_data.objects.filter(data_type = 'explain', keyword_id = ID, doc_date__range=(start_date, end_date),source_tag=1)
    # data_list = sorted(data_list, key=lambda x:x.doc_time, reverse=False)
    if len(origin_data_list)>1000:
        origin_data_list=origin_data_list[0:1000]
    if len(data_list)>1000:
        print('old len = ',len(data_list))
        data_list = data_list[0:1000]
        print('new len = ',len(data_list))
    
    context = {
        'id':ID,
        'title':title,
        'elements': elements,
        'origin_data_list':origin_data_list,
        'data_list':data_list,
    }
    data={
        'date_list':[],
        'total_list':[],
        'origin_total_list':[],
        'rate_list':[],
        'alias':[],
    }
    l=explain_charts_data.objects.filter(explain_id=ID)
    for item in l:
        data['date_list'].append(item.date.strftime("%Y/%m/%d"))
        data['total_list'].append(item.total)
        data['origin_total_list'].append(item.origin_count)
        data['rate_list'].append(item.rate)
    alias_list = alias.objects.filter(data_type='explain',keyword_id=myid)
    if len(alias_list)>0:
        for item in alias_list:
            data['alias'].append(item.alias_name)
    return render(request, 'regulation/explain_element_list.html', {'context': context,'Data': json.dumps(data)})





def weibo_display(request, param):
    l = param.split('&')
    category = l[0]
    keyword_name = l[1]
    element_id = l[2]
    if category =='law':
        # law_ful_name = l[1]
        # print('law_ful_name = ',law_ful_name)
        target = multi_version_law.objects.filter(ful_name = keyword_name)
        keyword_name = target[0].ful_name
        keyword_id = target[0].law_id
        print('keyword_name = ',keyword_name)
        print('keyword_id = ',keyword_id)
    else:
        target = explain.objects.filter(explain_name = keyword_name)
        keyword_id = target[0].explain_id

    
    if l[3].isdigit():
        page = int(l[3])-1
    else:
        page = 0

    temp_list = matched_clause_data.objects.filter(data_type = category,keyword_id = keyword_id, element_id = element_id)
    print('len of temp_list = ',len(temp_list))
    if category == 'law':
        rl = multi_version_law_clause.objects.filter(ful_name = keyword_name, clause_id = element_id)
        element_name = rl[0].clause_name
        content = rl[0].content
        print('element name = ',element_name)
        # element_name = law_clause.objects.filter(law_name = keyword_name, clause_id = element_id)[0].clause_name
    elif category == 'explain':
        rl = explain_element.objects.filter(explain_name = keyword_name, element_id = element_id)
        element_name = rl[0].element_name
        content = rl[0].content
    # keyword_id = temp_list[0].keyword_id.

    context = {
        'content':content,
        'keyword_name':keyword_name,
        'element_name':element_name,
        'data_list':[],
        'current_page':1,
        'total_num':1000,
    }

    data = {
        'alias':[],
        'digit_element':[],
        'hemi_url':category+'&'+keyword_name+'&'+element_id+'&',
    }

    alias_list = alias.objects.filter(formal_name=l[1])
    if len(alias_list)>0:
        for item in alias_list:
            data['alias'].append(item.alias_name)


    obj = re.compile('第(.{1,5})条')
    find_list = obj.findall(element_name)
    if len(find_list)>0:
        d = chinese2digits(find_list[0])
        digit_element = '第' + str(d) + '条'
        print(digit_element)
        data['digit_element'] = digit_element

    if len(temp_list)>0:
        if len(temp_list)>1000:
            begin = page*1000
            end = (page+1)*1000 if (page+1)*1000 <= len(temp_list) else len(temp_list)
            if begin < len(temp_list):
                context['data_list']=temp_list[begin:end]
                context['current_page']= (page+1)
            else:
                context['data_list']=temp_list[len(temp_list)-1000:len(temp_list)]
                context['current_page']=(len(temp_list)//1000)+1
            context['total_num']=len(temp_list)
            return render(request, 'regulation/weibo_display_multipage.html', {'context': context,'Data': json.dumps(data)})
        else:
            context['data_list'] = temp_list
            return render(request, 'regulation/weibo_display.html', {'context': context,'Data': json.dumps(data)})
    else:
        return render(request, 'regulation/weibo_display.html', {'context':context, 'Data': json.dumps(data)})

