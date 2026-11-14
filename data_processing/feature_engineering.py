import pandas as pd

def add_features(df):
    df = df.copy()
    df["amount_to_balance"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
    df["risk_exposure"] = df["merchant_risk_score"] * df["amount_usd"]
    df["risk_count"] = df[["is_foreign_transaction", "is_new_merchant", "used_vpn",
                           "ip_country_mismatch", "billing_shipping_mismatch",
                           "is_ai_generated_scam_attempt"]].sum(axis=1)
    df["cvv_scam"] = df["cvv_retry_count"] * df["is_ai_generated_scam_attempt"]
    return df
