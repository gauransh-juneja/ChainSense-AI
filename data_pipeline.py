import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

# Paths
DATA_FILE = "data/DataCoSupplyChainDataset.csv"
DB_FILE = "data/supply_chain.db"

def load_data():
    """Load CSV into DataFrame"""
    df = pd.read_csv(DATA_FILE, encoding='latin1')  # Kaggle file uses latin1 encoding
    print(f"✅ Loaded {len(df)} rows from {DATA_FILE}")
    return df

def clean_data(df):
    """Basic cleaning of dataset"""
    # Remove unnamed columns if any
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing numeric values with 0
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Fill missing strings with "Unknown"
    str_cols = df.select_dtypes(include=['object']).columns
    df[str_cols] = df[str_cols].fillna("Unknown")

    print(f"✅ Cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def store_data(df):
    """Store DataFrame into SQLite DB"""
    engine = create_engine(f"sqlite:///{DB_FILE}", echo=False)
    df.to_sql("supply_chain", con=engine, if_exists="replace", index=False)
    print(f"✅ Data stored in {DB_FILE} (table: supply_chain)")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    
    df = load_data()
    df = clean_data(df)
    store_data(df)
