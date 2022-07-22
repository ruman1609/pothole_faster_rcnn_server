from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from . import views

app_name = "predict"
urlpatterns = [
    path("", views.index, name="dashboard"),
    path("predict", csrf_exempt(views.PredictionAPI.as_view()), name="predict"),
]
