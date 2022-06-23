from django.urls import path
from . import views

app_name = "home"
urlpatterns = [
    path("", views.index, name = "index"), 
    path("eda", views.eda, name = "eda"), 
    path("calculate", views.calculate, name = "calculate")
]