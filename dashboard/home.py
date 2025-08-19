import streamlit as st

# Import your dashboards
from demand_forecasting import run_demand_forecasting
from sales_anomalies import run_sales_anomalies
from nlp_query_dashboard import run_nlp_query


def main():
    st.set_page_config(page_title="ChainSense-AI", layout="wide")

    # Sidebar Navigation
    st.sidebar.title("📊 Modules")
    page = st.sidebar.radio(
        "Go to",
        ("🏠 Home", "📈 Demand Forecasting", "⚠️ Sales Anomalies", "💬 NLP Query")
    )

    # Page Routing
    if page == "🏠 Home":
        st.title("📦 ChainSense-AI")
        st.markdown("""
        Welcome to **ChainSense-AI**, your AI-powered dashboard for:

        - 📈 Demand forecasting  
        - ⚠️ Sales anomaly detection  
        - 💬 Natural prompt-driven data insights  

        👉 Use the sidebar to explore modules.
        """)
    
    elif page == "📈 Demand Forecasting":
        run_demand_forecasting()
    
    elif page == "⚠️ Sales Anomalies":
        run_sales_anomalies()
    
    elif page == "💬 NLP Query":
        run_nlp_query()


if __name__ == "__main__":
    main()
