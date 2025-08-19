def run_demand_forecasting():
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st

    st.title("📈 Demand Forecasting")

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
        st.error("⚠️ Required column missing in dataset")
        st.stop()

    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")

    product_list = df["Product Name"].dropna().unique().tolist()
    product = st.selectbox("Select a product:", product_list)

    product_df = df[df["Product Name"] == product]
    sales_trend = (
        product_df.groupby(product_df["order_date"].dt.to_period("M"))["Order Item Quantity"]
        .sum()
        .reset_index()
    )
    sales_trend["order_date"] = sales_trend["order_date"].dt.to_timestamp()

    st.subheader(f"📊 Historical Demand for {product}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sales_trend["order_date"], sales_trend["Order Item Quantity"], marker="o")
    st.pyplot(fig)
