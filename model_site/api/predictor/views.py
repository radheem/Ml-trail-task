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
            # parsing the list for input features 
            var = str(var).split(",")
            # converting list to numpy array to use as input
            x = np.array(var).astype(np.float)
            # reshaping input to desirable shape
            x = x.reshape(1,4)
            # passing features into model
            prediction = PredictorConfig.model.predict(x)
            
            # build response
            response = {'y': list(prediction)}
            # return response
            return JsonResponse(response)
