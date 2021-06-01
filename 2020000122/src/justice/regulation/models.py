from django.db import models

# Create your models here.
from django.db import models
# 导入内建的User模型。
from django.contrib.auth.models import User
# timezone 用于处理时间相关事务。
from django.utils import timezone



class law_charts_data(models.Model):
    law_id=models.PositiveIntegerField(default = 0)
    law_name=models.CharField(max_length = 100)
    date= models.DateField()
    total=models.PositiveIntegerField(default = 0)
    origin_count = models.PositiveIntegerField(default=0)
    rate = models.FloatField(default= 1.00)
    class Meta:
        unique_together=("law_id","date")


class explain_charts_data(models.Model):
    explain_id=models.PositiveIntegerField(default = 0)
    explain_name=models.CharField(max_length = 100)
    date=models.DateField()
    total=models.PositiveIntegerField(default = 0)
    origin_count = models.PositiveIntegerField(default=0)
    rate = models.FloatField(default= 1.00)
    class Meta:
        unique_together=("explain_id","date")

class timestamp(models.Model):
    year=models.PositiveIntegerField(default = 2019)
    month=models.PositiveIntegerField(default = 1)
    day=models.PositiveIntegerField(default = 1)

class alias(models.Model):
    data_type = models.CharField(max_length = 20, default = "Unknown")
    keyword_id = models.PositiveIntegerField(default = 0)
    formal_name = models.CharField(max_length = 100)
    alias_name = models.CharField(max_length = 100)
    
class stopword(models.Model):
    keyword_name = models.CharField(max_length = 100)

class multi_version_law(models.Model):
    law_id = models.PositiveIntegerField(default = 0,primary_key=True)
    law_name = models.CharField(max_length = 100)
    ful_name = models.CharField(max_length = 100)
    timeliness = models.CharField(max_length = 20)
    department = models.CharField(max_length = 20)
    efficacyLevel = models.CharField(max_length = 20)
    releaseDate = models.BigIntegerField()
    effectiveDate = models.BigIntegerField()
    legalCategories = models.CharField(max_length = 150)
    content = models.TextField()
    total = models.PositiveIntegerField(default = 0)
    origin_count = models.PositiveIntegerField(default=0)


class multi_version_law_clause(models.Model):
    law_id = models.PositiveIntegerField(default = 0)
    clause_id = models.PositiveIntegerField(default = 0)
    ful_name = models.CharField(max_length = 100)
    law_name = models.CharField(max_length = 100)
    clause_name = models.CharField(max_length = 20)
    content = models.TextField()
    total = models.PositiveIntegerField(default = 0)
    origin_count = models.PositiveIntegerField(default=0)
    feature_words = models.TextField(default="/")
    spacial_words = models.TextField(default="/a=0.1/")
    class Meta:
        unique_together=("law_id","clause_id")


class law(models.Model):
    law_id = models.PositiveIntegerField(default = 0, primary_key = True)
    law_name = models.CharField(max_length = 100)
    # legalNumber = models.CharField(max_length = 50)
    timeliness = models.CharField(max_length = 20)
    department = models.CharField(max_length = 20)
    efficacyLevel = models.CharField(max_length = 20)
    releaseDate = models.BigIntegerField()
    effectiveDate = models.BigIntegerField()
    legalCategories = models.CharField(max_length = 150)
    content = models.TextField()
    total = models.PositiveIntegerField(default = 0)
    origin_count = models.PositiveIntegerField(default=0)


class law_clause(models.Model):
    law_id = models.PositiveIntegerField(default = 0)
    clause_id = models.PositiveIntegerField(default = 0)
    law_name = models.CharField(max_length = 100)
    clause_name = models.CharField(max_length = 20)
    content = models.TextField()
    total = models.PositiveIntegerField(default = 0)
    origin_count = models.PositiveIntegerField(default=0)
    class Meta:
        unique_together=("law_id","clause_id")

class explain(models.Model):
    explain_id = models.PositiveIntegerField(default = 0, primary_key = True)
    explain_name = models.CharField(max_length = 100)
    # legalNumber = models.CharField(max_length = 50)
    timeliness = models.CharField(max_length = 20)
    department = models.CharField(max_length = 20)
    efficacyLevel = models.CharField(max_length = 20)
    releaseDate = models.BigIntegerField()
    effectiveDate = models.BigIntegerField()
    legalCategories = models.CharField(max_length = 150)
    content = models.TextField()
    total = models.PositiveIntegerField(default = 0)
    origin_count = models.PositiveIntegerField(default=0)


class explain_element(models.Model):
    explain_id = models.PositiveIntegerField(default = 0)
    element_id = models.PositiveIntegerField(default = 0)
    explain_name =  models.CharField(max_length = 100)
    element_name =  models.CharField(max_length = 20)
    content = models.TextField()
    total = models.PositiveIntegerField(default = 0)
    origin_count = models.PositiveIntegerField(default=0)
    class Meta:
        unique_together=("explain_id","element_id")

class random_selected_data(models.Model):
    weibo_link = models.URLField(default='#')
    weibo_id = models.BigIntegerField(default = 0)
    tou_xiang = models.URLField()
    author_name = models.CharField(max_length = 30)
    doc_text = models.TextField()
    weibo_source = models.TextField()
    doc_date = models.DateField()

class nn_random_data(models.Model):
    weibo_link = models.URLField(default='#')
    weibo_id = models.BigIntegerField(default = 0)
    tou_xiang = models.URLField()
    author_name = models.CharField(max_length = 30)
    doc_text = models.TextField()
    weibo_source = models.TextField()
    doc_date = models.DateField()


class nn_label_data(models.Model):
    weibo_id = models.BigIntegerField(default = 0)
    belong_to_law = models.CharField(max_length = 70)
    belong_to_clause = models.CharField(max_length = 30)

class nn_fine_grain_training_data(models.Model):
    weibo_id = models.BigIntegerField(default = 0)
    belong_to_law = models.CharField(max_length = 70)
    belong_to_clause = models.CharField(max_length = 30)
    law_id = models.PositiveIntegerField(default = 0)
    clause_id = models.PositiveIntegerField(default = 0)
    content = models.TextField(default='nothing')
    weibo_content = models.TextField(default='nothing')
    label = models.PositiveIntegerField(default = 1)


class nn_auto_label_data(models.Model):
    weibo_id = models.BigIntegerField(default = 0)
    belong_to_law = models.CharField(max_length = 70)
    belong_to_clause = models.CharField(max_length = 30)
    law_id = models.PositiveIntegerField(default = 0)
    clause_id = models.PositiveIntegerField(default = 0)
    content = models.TextField(default='nothing')
    weibo_content = models.TextField(default='nothing')
    label = models.PositiveIntegerField(default = 1)

class real_law_data(models.Model):
    weibo_id = models.BigIntegerField(default = 0)
    belong_to_law = models.CharField(max_length = 70)

class real_clause_data(models.Model):
    weibo_id = models.BigIntegerField(default = 0)
    belong_to_law = models.CharField(max_length = 70)
    belong_to_clause = models.CharField(max_length = 30)

class judge_law_data(models.Model):
    weibo_id = models.BigIntegerField(default = 0)
    belong_to_law = models.CharField(max_length = 70)

class judge_clause_data(models.Model):
    weibo_id = models.BigIntegerField(default = 0)
    belong_to_law = models.CharField(max_length = 70)
    belong_to_clause = models.CharField(max_length = 30)



class solr_weibo_data(models.Model):
    weibo_id = models.BigIntegerField(default = 0,primary_key=True)
    law_process = models.IntegerField(default=0)
    clause_process = models.IntegerField(default=0)
    weibo_link=models.URLField(default="https://baidu.com")
    user_type = models.CharField(max_length = 20, default='unknown')# type
    author_name = models.CharField(max_length = 30, default='unknown')
    tou_xiang = models.URLField(default='https://baidu.com')
    doc_time = models.BigIntegerField(default = 0)
    doc_date = models.DateField(default='2019-01-01')
    doc_text = models.TextField(default='nothing')
    weibo_source = models.TextField(default='nothing')
    opinion = models.IntegerField(default = 0)


#存放法律与司法解释的数据
class matched_law_data(models.Model):
    weibo_id=models.BigIntegerField(default=0)
    keyword_id=models.PositiveIntegerField(default=0)
    keyword_name=models.CharField(max_length=100, default = "Unknown")
    weibo_link=models.URLField(default="https://baidu.com")
    data_type = models.CharField(max_length = 20, default = "Unknown")
    user_type = models.CharField(max_length = 20,default="Unknown")# type
    author_name = models.CharField(max_length = 30,default="Unknown")
    tou_xiang = models.URLField(default="https://baidu.com")
    doc_time = models.BigIntegerField(default = 0)
    doc_date = models.DateField(default="2019-01-01")
    doc_text = models.TextField(default="None")
    weibo_source = models.TextField(default="None")
    opinion = models.IntegerField(default = 0)
    origin_tag = models.IntegerField(default = 0)
    source_tag = models.IntegerField(default = 0)
    class Meta:
        unique_together=("weibo_id","keyword_id","data_type")


class matched_clause_data(models.Model):
    weibo_id = models.BigIntegerField(default = 0)
    keyword_id = models.PositiveIntegerField(default = 0)
    element_id = models.PositiveIntegerField(default = 0)
    keyword_name = models.CharField(max_length = 100, default = "Unknown")
    element_name = models.CharField(max_length = 20,  default = "Unknown")
    weibo_link=models.URLField(default="https://baidu.com")
    data_type = models.CharField(max_length = 20, default = "Unknown")#需要有data_type,否则keyword_id会冲突
    user_type = models.CharField(max_length = 20,default="Unknown")# type
    author_name = models.CharField(max_length = 30,default="Unknown")
    tou_xiang = models.URLField(default="https://baidu.com")
    doc_time = models.BigIntegerField(default = 0)
    doc_date = models.DateField(default="2019-01-01")
    doc_text = models.TextField(default="None")
    weibo_source = models.TextField(default="None")
    opinion = models.IntegerField(default = 0)
    origin_tag = models.IntegerField(default = 0)
    source_tag = models.IntegerField(default = 0)
    class Meta:
        unique_together=("weibo_id","keyword_id","element_id","data_type")