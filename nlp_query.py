import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

# Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Please set it in .env file.")

DB_FILE = "data/supply_chain.db"

# Connect to SQLite
engine = create_engine(f"sqlite:///{DB_FILE}")

def init_llm():
    """Initialize Groq LLM"""
    return ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it", streaming=False)

def init_db():
    """Initialize SQLite connection"""
    return SQLDatabase(engine)

def run_nlp_query(question):
    """Run NL query and return results as DataFrame"""
    llm = init_llm()
    db = init_db()

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    try:
        # Let the agent generate SQL
        sql_query = agent.run(f"Write only the SQL query for: {question}")
        
        print(f"\n📝 Generated SQL:\n{sql_query}\n")

        # Execute the SQL and return as DataFrame
        df = pd.read_sql(sql_query, con=engine)

        # Save query + result count in history table
        save_query_history(question, sql_query, len(df))

        return df

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return pd.DataFrame()

def save_query_history(user_q, sql_q, result_count):
    """Save query history in SQLite"""
    with engine.connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS query_history (user_query TEXT, sql_query TEXT, result_count INT)"
        )
        conn.execute(
            "INSERT INTO query_history (user_query, sql_query, result_count) VALUES (?, ?, ?)",
            (user_q, sql_q, result_count)
        )

if __name__ == "__main__":
    while True:
        user_q = input("\nAsk your supply chain question (or type 'exit'): ")
        if user_q.lower() == "exit":
            break
        df = run_nlp_query(user_q)
        if not df.empty:
            print("\n📊 Query Results:\n", df.head(10).to_string(index=False))
        else:
            print("⚠️ No results found.")
