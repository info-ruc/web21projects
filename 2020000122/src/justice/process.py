#!/usr/bin/env python
#coding:utf-8
import json
import os
import sys
import re
import datetime
import random
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "justice.settings")
if django.VERSION >= (1, 7):#自动判断版本
    django.setup()

from regulation import CompareMatch
from regulation import CombineMatch
from regulation import Match
from regulation import Data
from regulation import DB
from regulation import autoRenew
from regulation import multi_core_test
from regulation import gen_label_data
import DFA_NN_model
from regulation.models import judge_clause_data
def others():
    #匹配完成后,去掉法律法条和司法解释的()[]后缀
    # trim_title()
    #统计微博热度
    # tongji()
    # test_date()
    # test_match()
    
    # test()
    # l = ['一条至第十','一条，第二','三节等法律','8、9','四条、第十','四条、第十']
    # l = recheck(l)
    # print('l = ', l)

    # rename_file()
    # join()

    #添加微博链接
    # add_url()

    #添加别名到数据库
    # add_alias()
    print('others done!')




def data_process():
    # DB.delete_db()
    # print('delete done!')


    #从json文件中读取法律法条和司法解释信息到数据库
    # Data.lawdata2db()
    # Data.explaindata2db()
    
    
    #生成查询命令文件
    # Data.gen_cmd()
    # Data.generate_regulation_query_cmd()
    # Data.generate_explain_query_cmd()


    #导入查询到的微博数据，从读取文件到weibodata数据库
    # Data.weibodata2db(1)
    # Data.weibodata2db(2)

    print('data process done!')


def weibo_match():
    #匹配法律级别的微博数据,从weibodata到matched_law_data
    # Match.match_law_data()
    #匹配法条级别的微博数据,从weibodata到matched_clause_data
    # Match.match_clause_data()


    #从原始微博数据库weibodata中随机抽取200条微博到新的数据库，准备标注数据
    # select_random_weibo(200)
    #标注数据
    #匹配测试数据
    # Match.law_test_match()
    # Match.clause_test_match()

    #计算准确度、F1
    # Match.calculate_F1()
    print('weibo_match done!')

def total_db_renew():
    #改时间
    
    data_file_path='/home/zhan_su/original_tag_dir/project/data/'
    cmd_file_path='/home/zhan_su/original_tag_dir/project/judicial/solr/search_cmd.txt'
    solr_shell_path='/home/zhan_su/original_tag_dir/project/judicial/solr/bin/solr-shell'

    # duration, T =autoRenew.gen_cmd(data_file_path,cmd_file_path)
    # autoRenew.search(cmd_file_path,solr_shell_path)
    duration = '20190101_20200305'
    T = (2019,1,1,2020,3,5)
    # search_result_dir = autoRenew.unzip_file(data_file_path,duration)
    search_result_dir = data_file_path + duration +'/'
    # autoRenew.search_result_to_db(search_result_dir)
    
    #flag==0为未匹配的数据
    #匹配法律级别的微博数据,从weibodata到matched_law_data
    # Match.match_law_data()
    #匹配法条级别的微博数据,从weibodata到matched_clause_data
    Match.match_clause_data()
    
    #统计
    DB.tongji(T[0],T[1],T[2],T[3],T[4],T[5])
    print("renew completed!")
    


if __name__ == '__main__':
    # data_process()
    # weibo_match()
    '''
    data_file_path='/home/zhan_su/original_tag_dir/project/data/'
    cmd_file_path='/home/zhan_su/original_tag_dir/project/judicial/solr/search_cmd.txt'
    solr_shell_path='/home/zhan_su/original_tag_dir/project/judicial/solr/bin/solr-shell'
    autoRenew.gen_cmd(data_file_path,cmd_file_path)
    '''
    # autoRenew.search(cmd_file_path, solr_shell_path) 
    # multi_core_test.try_multi_core()
    # Match.test_trim_shuming()

    
    #匹配法律级别的微博数据,从weibodata到matched_law_data
    # Match.match_law_data()
    #匹配法条级别的微博数据,从weibodata到matched_clause_data
    # Match.match_clause_data()
    # DB.tongji(2019,1,1,2020,3,5)
    
    # law_clause.objects.filter(law_id=595,law_name="中华人民共和国刑法").delete()
    # law.objects.filter(law_id=595,law_name="中华人民共和国刑法").delete()

    # DB.add_law(597,"中华人民共和国土地管理法","中华人民共和国土地管理法(2019修正)","现行有效","全国人民代表大会","法律")
    # DB.add_clause_from_text(597,"中华人民共和国土地管理法","中华人民共和国土地管理法(2019修正)")
    
    # Match.add_new_feature_words()
    # Match.law_test_match()
    # Match.clause_test_match()
    # Match.calculate_F1()

    # Match.test_dtw()
    # DB.delete_db()
    
    # Match.test_dtw()
    
    # Match.add_feature_words()
    # Match.match_clause_data()
    # total_db_renew()
    # DB.tongji(2019,1,1,2020,3,5)
    # gen_label_data.select_data_from_solr(2000) 
    # gen_label_data.select_data_from_solr(2000)
    # Match.auto_mark_nn_data()
    # gen_label_data.gen_test_data()
    # DB.delete_db()
    # total_db_renew()
    # Match.migrate_fine_grain_data()
    # gen_label_data.gen_fine_train_data()
    # DFA_NN_model.load_arci_model()

    
    # Match.clause_test_match()
    # Match.add_feature_words()

    # CombineMatch.get_fine_training_data()
    # CombineMatch.gen_train_data_file()
    # CombineMatch.shuffle()

    # Match.calculate_F1()
    
    # CombineMatch.split_train_test_set()
    # CombineMatch.clause_test_match()
    # l = judge_clause_data.objects.all()

    # CompareMatch.clause_test_match()
    # CompareMatch.calculate_F1()
    
    # CompareMatch.buchong()
    # CompareMatch.kuochong()

    print('done!')

