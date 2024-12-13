"""
URL configuration for hearing_project project.

The urlpatterns list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from userapp import views as user_views
from adminapp import views as admin_views
from mainapp import views as mainapp_views

# URLS
urlpatterns = [
    # Main_Urls
    path('admin/', admin.site.urls),
    path('',mainapp_views.index,name='index'),
    path('about-us',mainapp_views.about_us,name='about_us'),
    path('user-login',mainapp_views.user_login,name='user_login'),
    path('admin-login',mainapp_views.admin_login,name='admin_login'),
    path('contact-us',mainapp_views.contact_us,name='contact_us'),
    path('register',mainapp_views.register,name='register'),
    path('otp',mainapp_views.otp,name='otp'),

    #User Views
    path('user-dashboard',user_views.user_dashboard,name='user_dashboard'),
    path('user-profile',user_views.user_profile,name='user_profile'),
    path('prediction',user_views.Classification,name='Classification'),
    path('prediction-result',user_views.Classification_result,name='Classification_result'),
    path('user-feedback',user_views.user_feedback,name='user_feedback'),
    path('user-logout',user_views.user_logout,name='user_logout'),


    #URLS_admin
    
    path('admin-dashboard',admin_views.admin_dashboard,name='admin_dashboard'),
   

    path('pending-users',admin_views.pending_users,name='pending_users'),
    path('all-users', admin_views.all_users, name='all_users'),
    path('delete-user/<int:user_id>/', admin_views.delete_user, name='delete_user'),
    path('accept-user/<int:id>', admin_views.accept_user, name = 'accept_user'),
    path('reject-user/<int:id>', admin_views.reject_user, name = 'reject'),
    path('change-status/<int:id>', admin_views.change_status, name = 'change_status'),
    path('adminlogout',admin_views.adminlogout, name='adminlogout'),

    # path('admin-feedback',admin_views.admin_feedback,name='admin_feedback'),
    # path('sentiment-analysis', admin_views.sentiment_analysis, name = 'sentiment_analysis'),
    # path('sentiment-analysis-graph',admin_views.sentiment_analysis_graph,name='sentiment_analysis_graph'),
    # path('comparision-graph',admin_views.comparision_graph,name='comparision_graph'),
    # path('data_exploration/', admin_views.data_exploration, name='data_exploration'),
    # path('data_preprocessing/', admin_views.data_preprocessing, name='data_preprocessing'),
    path('upload/', admin_views.upload_dataset, name = 'upload_dataset'),
    path('delete-dataset/<int:id>', admin_views.delete_dataset, name = 'delete_dataset'),
    path('view/', admin_views.view_dataset, name='view_dataset'),
    path('view_view/', admin_views.view_view_dataset, name = "view_view_dataset"),
    
    # path('XGBOOST', admin_views.XGBOOST, name='XGBOOST'),
    # path('XGBOOST_btn', admin_views.XGBOOST_btn, name='XGBOOST_btn'),
    
    path('ADABoost', admin_views.ADABoost, name='ADABoost'),
    path('ADABoost_btn', admin_views.ADABoost_btn, name='ADABoost_btn'),

    path('upload/', admin_views.upload_dataset, name='upload_dataset'),
    path('delete-dataset/<int:id>/', admin_views.delete_dataset, name='delete_dataset'),
    path('view/', admin_views.view_dataset, name='view_dataset'),
    path('view_view/', admin_views.view_view_dataset, name='view_view_dataset'),
    
    path('RandomForest/', admin_views.Random_Forest, name = 'Random_Forest'),
    path('RandomForest_btn', admin_views.RandomForest_btn, name = 'RandomForest_btn'),
    
    path('Gradientboosting/', admin_views.Gradient_boosting, name = 'Gradient_boosting'),
    path('Gradient_boosting_btn', admin_views.Gradient_boosting_btn, name = 'Gradient_boosting_btn'),
    
    path('Decision_tree/', admin_views.Decision_tree, name = 'Decision_tree'),
    path('Decision_tree_btn', admin_views.Decision_tree_btn, name = 'Decision_tree_btn'),

    path('XG_Boost/', admin_views.XG_Boost, name = 'XG_Boost'),
    path('XG_Boost_btn', admin_views.XG_Boost_btn, name = 'XG_Boost_btn'),

    path('Logistic_Regression', admin_views.Logistic_Regression, name = 'Logistic_Regression'),
    path('Logistic_Regression_btn', admin_views.Logistic_Regression_btn, name = 'Logistic_Regression_btn'),

    path('Knn', admin_views.Knn, name = 'Knn'),
    path('Knn_btn', admin_views.Knn_btn, name = 'Knn_btn'),

    path('Exploration', admin_views.Exploration, name = 'Exploration'),


    path('Comparision', admin_views.Comparision, name = 'Comparision'),

    path('admin_Feedback', admin_views.admin_Feedback, name = 'admin_Feedback'),
    path('admin_Sentimet_analysis', admin_views.admin_Sentimet_analysis, name = 'admin_Sentimet_analysis'),
    path('admin_Sentimet_analysis_graph', admin_views.admin_Sentimet_analysis_graph, name = 'admin_Sentimet_analysis_graph'),


    
    
    # path('Extra_tree', admin_views.Extra_tree, name = 'Extra_tree'),
    # path('Extra_tree_btn/', admin_views.Extra_tree_btn, name = 'Extra_tree_btn')
    # path('admin-feedback',admin_views.admin_feedback,name='admin_feedback'),
    # path('sentiment-analysis', admin_views.sentiment_analysis, name = 'sentiment_analysis'),
    # path('sentiment-analysis-graph',admin_views.sentiment_analysis_graph,name='sentiment_analysis_graph'),
    # path('comparision-graph',admin_views.comparision_graph,name='comparision_graph'),

]   + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)



