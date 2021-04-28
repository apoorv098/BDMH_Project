from django.urls import path,include
from first_component import views

# Template Tagging
app_name = 'first_component'

urlpatterns = [
    path('about/', views.about, name = 'about'),
    path('predict/', views.predict, name = 'predict'),
    path('result/', views.result, name = 'result'),
]

