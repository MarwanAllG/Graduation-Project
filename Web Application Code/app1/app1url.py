from django.urls import path
from app1 import views
urlpatterns = [
    path('',views.home,name='home'),
    path('dashboard', views.index, name='index'),
    path('motor',views.motor,name='motor')
]