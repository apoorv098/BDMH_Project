from django.shortcuts import render, redirect
from .forms import Predict_Form
from first_component import data_provider, lime_explainer
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
import numpy as np

# Create your views here.
def index(request):
    return render(request, 'first_component/index.html')


def result(request):
    return render(request, 'first_component/result.html')

 




def predict(request):
    form = Predict_Form(data=request.POST)
    predicted = False
    predictions = {}
    
    if request.method == 'POST':
        form = Predict_Form(data=request.POST)

        if form.is_valid():
                form.save(commit = True)
                age = request.POST.get('age')
                sex = request.POST.get('sex')
                cp =  request.POST.get('cp')
                resting_bp = request.POST.get('resting_bp')
                serum_cholesterol = request.POST.get('serum_cholesterol')
                fasting_blood_sugar = request.POST.get('fasting_blood_sugar')
                resting_ecg = request.POST.get('resting_ecg')
                max_heart_rate = request.POST.get('max_heart_rate')
                exercise_induced_angina = request.POST.get('exercise_induced_angina')
                st_depression = request.POST.get('st_depression')
                st_slope = request.POST.get('st_slope')
                number_of_vessels = request.POST.get('number_of_vessels')
                thallium_scan_results = request.POST.get('thallium_scan_results')
                print(age, sex, cp, resting_bp)

                features = [[age,sex,cp,resting_bp,serum_cholesterol,fasting_blood_sugar,resting_ecg,max_heart_rate,exercise_induced_angina,
                        st_depression,st_slope,number_of_vessels,thallium_scan_results]]
                
                
                features = np.asarray(features, dtype='float64')
                SVCClassifier,DecisionTreeClassifier,KNN,RF,MLP = data_provider.GetAllClassifiers()
                predictions = {'SVC': str(SVCClassifier.predict(features)[0]),
                'DecisionTree': str(DecisionTreeClassifier.predict(features)[0]),
                'KNN': str(KNN.predict(features)[0]),
                'RF': str(RF.predict(features)[0]),
                'MLP': str(MLP.predict(features)[0]),
                }

                pred = form.save(commit=False)

                
                l=[predictions['MLP'],predictions['SVC'], predictions['DecisionTree'], predictions['KNN'],predictions['RF']]
                count=l.count('1')

                result=False

                if count >=  3:
                        result=True
                        pred.num=1
                elif (count == 0):
                        pred.num=0

                
                predicted = True
                to_explain = False



                
                mlp_explainer = lime_explainer.MLPExplainer()
                exp_lime_mlp = mlp_explainer.explain(features[0])

                svm_explainer = lime_explainer.SVMExplainer()
                exp_lime_svm = svm_explainer.explain(features[0])

                knn_explainer = lime_explainer.KNNExplainer()
                exp_lime_knn = knn_explainer.explain(features[0])

                rf_explainer = lime_explainer.RFExplainer()
                exp_lime_rf = rf_explainer.explain(features[0])

                dt_explainer = lime_explainer.DecisionTreeExplainer()
                exp_lime_dt = dt_explainer.explain(features[0])


                to_explain = True
                exp_lime_mlp.save_to_file("static/first_component/explanations/explanation_mlp.html")
                exp_lime_svm.save_to_file("static/first_component/explanations/explanation_svm.html")
                exp_lime_knn.save_to_file("static/first_component/explanations/explanation_knn.html")
                exp_lime_rf.save_to_file("static/first_component/explanations/explanation_rf.html")
                exp_lime_dt.save_to_file("static/first_component/explanations/explanation_dt.html")

              



    if (predicted and to_explain):
        return render(request, 'first_component/result.html',
               {'predicted': predicted,'predictions':predictions,'result':result, 'exp_lime_mlp':exp_lime_mlp,
                 'exp_lime_svm':exp_lime_svm, 'exp_lime_knn':exp_lime_knn, 'exp_lime_rf':exp_lime_rf, 
                 'exp_lime_dt':exp_lime_dt})


    else:
        form = Predict_Form()

        return render(request, 'first_component/predict.html',
                      {'form': form,'predicted': predicted,'predictions':predictions})







