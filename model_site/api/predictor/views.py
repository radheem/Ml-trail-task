from django.shortcuts import render
from .apps import PredictorConfig
from django.http import JsonResponse
from rest_framework.views import APIView
import numpy as np
def home(request):
    return render(request,'home.html')
class call_model(APIView):
    def get(self,request):
        if request.method == 'GET':
            # get sound from request
            var1 = request.GET.get('MaxTemp')
            var2 = request.GET.get('MinTemp')
            var3 = request.GET.get('Temp9AM')
            var4 = request.GET.get('Temp3PM')
            
            # converting list to numpy array to use as input
            x = np.array([var1,var2,var3,var4]).astype(np.float)
            # reshaping input to desirable shape
            x = x.reshape(1,4)
            # passing features into model
            prediction = PredictorConfig.model.predict(x)
            
            # build response
            response = {'y': list(prediction)}
            # return response
            return JsonResponse(response)
