from django.shortcuts import render
from .apps import PredictorConfig
from django.http import JsonResponse
from rest_framework.views import APIView
import numpy as np
class call_model(APIView):
    def get(self,request):
        if request.method == 'GET':
            # get sound from request
            var = request.GET.get('var')
            x = np.array(var)
            x = x.reshape(1,-1)
            prediction = PredictorConfig.model.predict(x)
            # build response
            response = {'y': prediction}
            # return response
            return JsonResponse(response)
