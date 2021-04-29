from django.contrib import admin
from first_component.models import Predictions
from django import forms

# Register your models here.

class Prediction(admin.ModelAdmin):
    list_display=('age','sex','cp','resting_bp','serum_cholesterol','fasting_blood_sugar','resting_ecg','max_heart_rate','exercise_induced_angina','st_depression','st_slope','number_of_vessels','thallium_scan_results','predicted_on','num')

admin.site.register(Predictions,Prediction)