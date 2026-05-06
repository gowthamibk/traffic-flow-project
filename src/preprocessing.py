import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # Remove Total rows
    df = df[~df['State/UT/City'].str.contains("Total", na=False)]

    # Drop text columns
    df = df.drop(['Sl. No.', 'State/UT/City'], axis=1)

    # Fill missing values
    df = df.fillna(0)

    return df
