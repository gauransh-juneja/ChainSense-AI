import pandas as pd
from sqlalchemy import create_engine

DB_FILE = "data/supply_chain.db"

# Load data from SQLite
engine = create_engine(f"sqlite:///{DB_FILE}")
df = pd.read_sql("SELECT DISTINCT `Product Card Id`, `Product Name` FROM supply_chain", con=engine)

# Display unique products
print(f"✅ Found {len(df)} unique products:\n")
print(df.to_string(index=False))
