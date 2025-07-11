import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Load API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]
if not api_key:
    st.error("❌ OpenAI API key not found. Please add it to Streamlit secrets.")
    st.stop()

# Init LLM
llm = OpenAI(api_key=api_key, temperature=0)

# Page config
st.set_page_config(page_title="Queryfy", layout="wide")
st.title("📊 Queryfy - Chat with Your Data")

# File uploader
uploaded_file = st.file_uploader("📁 Upload a file (CSV, Excel, TSV, or JSON)", type=["csv", "xlsx", "tsv", "json"])

if uploaded_file is not None:
    try:
        # Detect format and read data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".tsv"):
            df = pd.read_csv(uploaded_file, sep='\t')
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.success("✅ File uploaded successfully!")

        # Show full data
        st.subheader("📄 Your Data")
        st.dataframe(df, use_container_width=True, height=600)

        # Chart builder
        st.subheader("📈 Build a Chart")
        chart_type = st.selectbox("Choose a chart type", ["None", "Line", "Bar", "Pie"])

        if chart_type != "None":
            x_axis = st.selectbox("🧭 Select X-axis", df.columns)
            y_axis = st.selectbox("🧮 Select Y-axis", df.columns)
            if st.button("Generate Chart"):
                fig = None
                if chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis)
                elif chart_type == "Pie":
                    fig = px.pie(df, names=x_axis, values=y_axis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # LangChain agent
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            max_execution_time=1600,
            max_iterations=1000,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True
        )

        st.subheader("💬 Chat With Your Data")
        st.markdown("💡 Example questions:")
        st.code("Total sales by category\nAverage revenue by month\nTop 5 products by profit")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display previous messages
        for user_msg, bot_msg in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(bot_msg)

        # Clean user query
        def clean_query(q):
            q = q.strip()
            if not q.endswith("?") and not q.lower().startswith("show"):
                q += "?"
            return f"You are a helpful data analyst. Based on the uploaded file, answer this: {q}"

        # Chat input
        user_input = st.chat_input("Ask something about your data...")

        if user_input:
            cleaned = user_input.strip()
            if not cleaned:
                st.warning("⚠️ Please enter a valid question.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        response = agent.run(clean_query(cleaned))
                    except Exception as e:
                        response = f"❌ Error: {e}"

                st.session_state.chat_history.append((user_input, response))
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
