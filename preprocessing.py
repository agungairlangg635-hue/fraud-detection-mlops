import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_features_target(df, target_column="Class"):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def create_features(df):
    df = df.copy()
    df["Amount_Log"] = df["Amount"].apply(lambda x: __import__("numpy").log1p(x))
    df["Hour"] = ((df["Time"] / 3600) % 24).astype(int)
    return df