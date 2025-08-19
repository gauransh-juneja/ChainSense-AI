# ChainSense-AI  

ChainSense-AI is an AI-powered **Supply Chain Analytics Dashboard** that helps businesses analyze, forecast, and detect anomalies in their operations.  
The system provides insights through **Demand Forecasting**, **Sales Anomaly Detection**, and **NLP-based Queries** — all in one unified dashboard.  

---

## 🚀 Features  

- 📊 **Demand Forecasting** – Predict future demand trends from supply chain datasets.  
- 📉 **Sales Anomaly Detection** – Identify unusual patterns or outliers in sales data.  
- 🤖 **NLP Query Assistant** – Ask natural language questions about the data (powered by LLMs).  
- 🖥 **Interactive Dashboard** – Navigate between modules seamlessly.  

---

## 📂 Project Structure  

ChainSense-AI/
│── dashboard/
│ ├── home.py # Main dashboard navigation
│ ├── demand_forecasting.py # Demand forecasting module
│ ├── sales_anomalies.py # Sales anomaly detection module
│ ├── nlp_query_dashboard.py # NLP query assistant module
│
│── data/
│ ├── DataCoSupplyChainDataset.csv # Main dataset
│ ├── DescriptionDataCoSupplyChain.csv # Dataset description
│ ├── supply_chain.db # Database file
│ ├── tokenized_access_logs.csv # Log file
│
│── requirements.txt # Python dependencies
│── ml_module.py # ML models utilities
│── data_pipeline.py # Data preprocessing pipeline
│── check_products.py # Product validation script
│── nlp_query.py # NLP query backend logic
│── .env # (User must create & add their API key here)
│── README.md # Project documentation

---
## ⚙️ Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/gauransh-juneja/ChainSense-AI.git
   cd ChainSense-AI
2. Create Virtual Environment:
   python -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate      # (Windows PowerShell)

4. Install Dependencies:
   pip install -r requirements.txt
---

## ▶️ Usage

1. Run the dashboard:
   streamlit run dashboard/home.py
2. Navigate between modules using the sidebar:
   Demand Forecasting 📊
   Sales Anomaly Detection 📉
   NLP Query Assistant 🤖
--
## Contibution
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

##📜 License

This project is licensed under the MIT License
.

You are free to use, modify, and distribute this project for personal or commercial purposes, provided that the copyright notice is included.
