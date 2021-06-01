# 引入path
from django.urls import path
from . import views

# 正在部署的应用的名称
app_name = 'regulation'

urlpatterns = [
    # 目前还没有urls
    path('show_5000_random_data/', views.show_5000_random_data, name='show_5000_random_data'),
    path('positive_data/', views.positive_data, name='positive_data'),
    path('negative_data/', views.negative_data, name='negative_data'),
    path('show_new_training_data/', views.show_new_training_data, name='show_new_training_data'),
    path('delete_nn_data/', views.delete_nn_data, name='delete_nn_data'),
    path('change_nn_data_label/', views.change_nn_data_label, name='change_nn_data_label'),
    path('show_fine_grain_training_data/', views.show_fine_grain_training_data, name='show_fine_grain_training_data'),
    path('show_nn_training_data/', views.show_nn_training_data, name='show_nn_training_data'),
    path('show_random_data/', views.show_random_data, name='show_random_data'),
    path('history_version_display/<int:id>/', views.history_version_display, name='history_version_display'),
    path('law_list/', views.law_list, name='law_list'),
    path('law_clause_list/<str:param>/', views.law_clause_list, name='law_clause_list'),
    path('explain_list/', views.explain_list, name='explain_list'),
    path('explain_element_list/<str:param>/', views.explain_element_list, name='explain_element_list'),
    path('weibo_display/<str:param>/', views.weibo_display, name='weibo_display'),
    path('dynamic_display/', views.dynamic_display, name='dynamic_display'),
]