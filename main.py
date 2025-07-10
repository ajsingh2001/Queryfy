import streamlit as st
import pandas as pd
import os

from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Load API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Check if API key is loaded
if not api_key:
    st.error("❌ OpenAI API key not found. Please add it to Streamlit secrets.")
    st.stop()

# Initialize OpenAI LLM
llm = OpenAI(api_key=api_key, temperature=0)

# Streamlit UI
st.set_page_config(page_title="Talk to your CSV or Excel")
st.header("📊 Ask Anything About Your Data")

# File uploader
csv_file = st.file_uploader("📁 Upload a CSV or Excel file", type=["csv", "xlsx"])

if csv_file is not None:
    try:
        # Read uploaded file
        if csv_file.name.endswith(".csv"):
            df = pd.read_csv(csv_file)
        elif csv_file.name.endswith(".xlsx"):
            df = pd.read_excel(csv_file)

        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        # Create LangChain agent with dangerous code execution enabled
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            max_execution_time=1600,
            max_iterations=1000,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True  # ⚠️ Required for code execution
        )

        # User query input
        query = st.text_input("💬 Ask a question about your file")

        if st.button("Submit", type="primary"):
            if query.strip():
                with st.spinner("Thinking..."):
                    response = agent.run(query)
                st.markdown("**🧠 Response:**")
                st.write(response)
            else:
                st.warning("⚠️ Please enter a question.")

    except Exception as e:
        st.error(f"❌ Error reading or processing file: {e}")
