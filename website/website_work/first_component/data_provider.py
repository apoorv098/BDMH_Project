import os
import pickle
import joblib

config = {
    'heart': {
        'SVC': 'ml_models/svc_model.pkl',
        'DecisionTree':'ml_models/decision_tree_model.pkl',
        'KNN':'ml_models/knn_model.pkl',
        'RF':'ml_models/random_forest_model.pkl',
        'MLP':'ml_models/mlp_model.pkl',
    }}


dir = os.path.dirname(__file__)
print(dir)
def GetJobLibFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return joblib.load(os.path.join(dir, filepath))
    else:
        print("file does not exit")

def GetPickleFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return pickle.load( open(os.path.join(dir, filepath), "rb" ) )
    return None


def GetAllClassifiers():
    return (GetSVCClassifier(),GetDecisionTreeClassifier(),GetKNN(),GetRF(),GetMLP())

def GetSVCClassifier():
    return GetJobLibFile(config['heart']['SVC'])

def GetDecisionTreeClassifier():
    return GetJobLibFile(config['heart']['DecisionTree'])

def GetKNN():
    return GetJobLibFile(config['heart']['KNN'])

def GetRF():
    return GetJobLibFile(config['heart']['RF'])

def GetMLP():
    return GetJobLibFile(config['heart']['MLP'])

