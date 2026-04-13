import pandas as pd

def load_fraud_data(transaction_path: str, identity_path: str):
    df_trans = pd.read_csv(transaction_path)
    df_id = pd.read_csv(identity_path)

    # Merge on TransactionID
    df = df_trans.merge(df_id, on="TransactionID", how="left")

    # Separate label
    y = df["isFraud"]
    X = df.drop(columns=["isFraud"])

    return X, y