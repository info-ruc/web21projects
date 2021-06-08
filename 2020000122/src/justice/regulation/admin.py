from django.contrib import admin

# 别忘了导入ArticlerPost
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
from .models import judge_law_data
from .models import real_law_data
from .models import judge_clause_data
from .models import real_clause_data
from .models import alias
from .models import stopword
from .models import timestamp
from .models import nn_random_data
from .models import nn_label_data
from .models import nn_auto_label_data
from .models import nn_fine_grain_training_data

# 注册ArticlePost到admin中
admin.site.register(solr_weibo_data)
admin.site.register(law)
admin.site.register(law_clause)
admin.site.register(multi_version_law)
admin.site.register(multi_version_law_clause)
admin.site.register(explain)
admin.site.register(explain_element)
admin.site.register(matched_clause_data)
admin.site.register(matched_law_data)
admin.site.register(law_charts_data)
admin.site.register(explain_charts_data)
admin.site.register(random_selected_data)
admin.site.register(judge_law_data)
admin.site.register(real_law_data)
admin.site.register(judge_clause_data)
admin.site.register(real_clause_data)
admin.site.register(alias)
admin.site.register(stopword)
admin.site.register(timestamp)
admin.site.register(nn_random_data)
admin.site.register(nn_label_data)
admin.site.register(nn_auto_label_data)
admin.site.register(nn_fine_grain_training_data)