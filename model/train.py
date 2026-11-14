import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from utils.config import RANDOM_STATE

class kfold():
    def __init__(self,n_split=5,
                  shuffle=True,
                  random_state=RANDOM_STATE):
        self.skf = StratifiedKFold(n_splits=n_split,
                                   shuffle=shuffle,
                                   random_state=random_state)
        
    def split(self,x,y):
        for train_idx,val_idx in self.skf.split(x,y):
            x_train,x_val = x.iloc[train_idx],x.iloc[val_idx]
            y_train,y_val = y.iloc[train_idx],y.iloc[val_idx]
            yield x_train, x_val, y_train, y_val

model = {
    "LGBMClassifier":LGBMClassifier(n_estimators=3000,
                                    learning_rate=0.05,
                                    num_leaves=4,
                                    max_depth=2,
                                    min_child_samples=70,
                                    verbose=-1,
                                    random_state=RANDOM_STATE)
}

def cv_engine(x, y, models, kf):
    results = []
    for name, model in models.items():
        fold_scores = []
        for x_tr, x_va, y_tr, y_va in kf.split(x, y):
            m = clone(model)                      # fresh, untrained copy for each fold
            m.fit(x_tr, y_tr)
            proba = m.predict_proba(x_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, proba))
        results.append({
            "model_name": name,
            "cv_mean": np.mean(fold_scores),
            "cv_std": np.std(fold_scores),
        })
        print(name, [round(s, 5) for s in fold_scores])
    return pd.DataFrame(results).sort_values("cv_mean", ascending=False)
