import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# ✅ Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

# ✅ Initialize Groq LLM
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    api_key=groq_api_key
)

# ✅ Connect to SQLite database
DB_PATH = "data/supply_chain.db"
if not os.path.exists(DB_PATH):
    st.error(f"❌ Database not found at {DB_PATH}. Please check your data folder.")
    st.stop()

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# ✅ Create SQL Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# ✅ Streamlit App
def run_nlp_query():
    st.title("💬 NLP Query Assistant")
    st.markdown("Ask natural language questions about your supply chain data:")

    query = st.text_input("🔍 Enter your query:")
    
    if query:
        with st.spinner("Thinking... 🤔"):
            try:
                response = agent_executor.run(query)
                st.success("✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
