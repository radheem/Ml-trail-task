from django.apps import AppConfig
from django.conf import settings
import os
from sklearn.externals import joblib

class PredictorConfig(AppConfig):
    # create path to models
    path1 = os.path.join(settings.MODELS, 'LinearReg.sav')
    model1 = joblib.load(path1)
    path2 = os.path.join(settings.MODELS, 'Lasso.sav')
    model2 = joblib.load(path2)
    path3 = os.path.join(settings.MODELS, 'Ridge.sav')
    model3 = joblib.load(path3)