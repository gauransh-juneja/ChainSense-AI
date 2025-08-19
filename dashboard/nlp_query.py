import streamlit as st
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page config
st.set_page_config(page_title="NLP Query", page_icon="💬")
st.title("💬 Natural Language Queries on Supply Chain Data")

# Connect to SQLite database
db = SQLDatabase.from_uri("sqlite:///data/supply_chain.db")

# Setup LLM (Groq API)
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=GROQ_API_KEY)

# Create SQL Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm, toolkit=toolkit, verbose=True
)

# User Input
user_query = st.text_input("Ask a question about the supply chain database:")

if user_query:
    try:
        with st.spinner("Thinking..."):
            result = agent_executor.run(user_query)

        st.success("Answer:")
        st.write(result)

        # Try to display results as table + chart
        try:
            conn = sqlite3.connect("data/supply_chain.db")

            # Extract SQL query if result looks like SQL
            if "SELECT" in result.upper():
                df = pd.read_sql_query(result, conn)

                if not df.empty:
                    st.subheader("📊 Query Results")
                    st.dataframe(df)

                    # Show chart if numeric data exists
                    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
                    if len(numeric_cols) > 0:
                        st.subheader("📈 Visualization")

                        # Default: show first numeric column
                        col_to_plot = st.selectbox("Choose column to plot:", numeric_cols)

                        fig, ax = plt.subplots()
                        if len(df) < 20:  # small dataset → bar chart
                            df.plot(x=df.columns[0], y=col_to_plot, kind="bar", ax=ax)
                        else:  # large dataset → line chart
                            df.plot(x=df.columns[0], y=col_to_plot, kind="line", ax=ax)

                        st.pyplot(fig)

            conn.close()
        except Exception as e:
            st.warning(f"Could not display table/chart: {str(e)}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
