from django.apps import AppConfig
from django.conf import settings
import os
from sklearn.externals import joblib

class PredictorConfig(AppConfig):
    # create path to models
    path = os.path.join(settings.MODELS, 'LinearReg.sav')
    model = joblib.load(path)