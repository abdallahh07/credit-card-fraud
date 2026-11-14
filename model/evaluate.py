from sklearn.metrics import roc_auc_score

def evaluate_model(model,x_test,y_test):
    y_pred = model.predict_proba(x_test)[:, 1]
    return roc_auc_score(y_test,y_pred)
