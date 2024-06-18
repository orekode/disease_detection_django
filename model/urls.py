from django.urls import path
from .views import submit

urlpatterns = [
    path('submit/', submit, name='submit'),
]
