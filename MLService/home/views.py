from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseNotFound, JsonResponse
from django.shortcuts import render
from django.template import loader
from pydantic import Json
from home.model.inference import get_model_response
import json 

# Create your views here.
def index(request):
    return render(request, "home/index.html")

def eda(request):
    return render(request, "home/eda.html")

def calculate(request):

    try:
        feature_dict = dict(request.POST)
    except ValueError as e:
        return {'error': str(e)}, 422

    if not feature_dict or feature_dict == "":
        return HttpResponseBadRequest("Body is empty")
    try:
        response = get_model_response(feature_dict)
    except Exception as e:
        return HttpResponseNotFound("An error has occured during inference time. Try later. " + str(e))
    return JsonResponse(response)