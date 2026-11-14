import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def x_y(train):
    x = train.drop(columns=["id","is_fraud"])
    y = train["is_fraud"]
    return x,y

def encoder(df):
    df = df.copy()
    bool_cols = df.select_dtypes(include = "bool").columns
    cat_cols = df.select_dtypes(include=["object", "str"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    
    return df
