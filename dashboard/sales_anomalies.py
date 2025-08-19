def run_sales_anomalies():
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st

    st.title("🚨 Sales Anomaly Detection")

    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")
        except FileNotFoundError:
            st.error("❌ Data file not found in data/ folder.")
            return pd.DataFrame()

    df = load_data()
    if df.empty:
        st.stop()

    if "order date (DateOrders)" not in df.columns:
        st.error("⚠️ Required column missing")
        st.stop()

    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")

    daily_sales = (
        df.groupby(df["order_date"].dt.date)["Order Item Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"order_date": "Date", "Order Item Quantity": "Total Sales"})
    )

    Q1, Q3 = daily_sales["Total Sales"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    daily_sales["Anomaly"] = daily_sales["Total Sales"].apply(
        lambda x: "Anomaly" if x < lower or x > upper else "Normal"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_sales["Date"], daily_sales["Total Sales"], marker="o")
    anomalies = daily_sales[daily_sales["Anomaly"] == "Anomaly"]
    ax.scatter(anomalies["Date"], anomalies["Total Sales"], color="red", label="Anomalies")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🚩 Detected Anomalies")
    st.dataframe(anomalies)
