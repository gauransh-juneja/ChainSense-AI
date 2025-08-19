import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
import plotly.express as px

DB_FILE = "data/supply_chain.db"

def load_data():
    """Load data from SQLite database"""
    engine = create_engine(f"sqlite:///{DB_FILE}")
    df = pd.read_sql("SELECT * FROM supply_chain", con=engine)
    return df

def forecast_demand(df, product_id, periods=30):
    """Forecast future sales for a given product"""
    product_df = df[df['Product Card Id'] == product_id].copy()

    # Group by date
    daily_sales = product_df.groupby('order date (DateOrders)')['Order Item Quantity'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])

    model = Prophet()
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def detect_anomalies(df):
    """Detect anomalies in sales data"""
    sales_df = df.groupby('order date (DateOrders)')['Order Item Quantity'].sum().reset_index()
    sales_df['order date (DateOrders)'] = pd.to_datetime(sales_df['order date (DateOrders)'])
    sales_df['day_of_year'] = sales_df['order date (DateOrders)'].dt.dayofyear

    iso = IsolationForest(contamination=0.02, random_state=42)
    sales_df['anomaly'] = iso.fit_predict(sales_df[['Order Item Quantity', 'day_of_year']])

    anomalies = sales_df[sales_df['anomaly'] == -1]
    return anomalies

def save_to_db(df, table_name):
    """Save DataFrame to SQLite table"""
    engine = create_engine(f"sqlite:///{DB_FILE}")
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    print(f"✅ Saved {len(df)} rows to table '{table_name}'")

def plot_forecast(forecast_df):
    """Plot forecast using Plotly"""
    fig = px.line(forecast_df, x='ds', y='yhat', title="Demand Forecast")
    fig.show()

def plot_anomalies(anomalies_df):
    """Plot anomalies using Plotly"""
    fig = px.scatter(anomalies_df, x='order date (DateOrders)', y='Order Item Quantity',
                     color='anomaly', title="Sales Anomalies")
    fig.show()

if __name__ == "__main__":
    df = load_data()

    # Example: pick first product and forecast 30 days
    example_product_id = df['Product Card Id'].iloc[0]

    forecast_df = forecast_demand(df, product_id=example_product_id, periods=30)
    anomalies_df = detect_anomalies(df)

    # Save results
    save_to_db(forecast_df, "forecast_results")
    save_to_db(anomalies_df, "sales_anomalies")

    # Visual checks
    plot_forecast(forecast_df)
    plot_anomalies(anomalies_df)
