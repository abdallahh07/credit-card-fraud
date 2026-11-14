from sklearn.metrics import roc_auc_score
from train import model

model = model()

def evaluate_model(model,x_test,y_test):
    y_pred = model(x_test)
    return roc_auc_score(y_test,y_pred)
