import tempfile
import csv
import streamlit as st
import pandas as pd
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Analyst Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .sub-header {
            color: #6b7280;
            font-size: 1rem;
            margin-top: 0.1rem;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: #f8f9fb;
            border: 1px solid #eaecef;
            border-radius: 10px;
            padding: 12px 16px;
        }
        .stButton>button {
            border-radius: 8px;
        }
        section[data-testid="stSidebar"] .stButton>button {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def preprocess_and_save(file):
    """Read the uploaded file, clean up dtypes, and cache the result to a temp CSV."""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file, na_values=["NA", "N/A", "missing"])
        else:
            return None, None, None

        # Parse dates and numeric columns
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df
    except Exception as e:
        return None, None, str(e)


@st.cache_resource(show_spinner=False)
def get_agent(api_key, temp_path):
    """Build (and cache) the DuckDB-backed data analyst agent for this file/key pair."""
    duckdb_tools = DuckDbTools()
    duckdb_tools.load_local_csv_to_table(path=temp_path, table="uploaded_data")

    return Agent(
        model=OpenAIChat(id="gpt-4o", api_key=api_key),
        tools=[duckdb_tools, PandasTools()],
        system_message=(
            "You are an expert data analyst. Use the 'uploaded_data' table to answer "
            "user queries. Generate SQL queries using DuckDB tools to solve the user's "
            "query. Provide clear and concise answers with the results."
        ),
        markdown=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("openai_key", "")
st.session_state.setdefault("pending_query", None)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    openai_key = st.text_input(
        "OpenAI API key", type="password", value=st.session_state.openai_key
    )
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("API key saved", icon="✅")
    else:
        st.warning("Enter your OpenAI API key to get started.")

    st.divider()
    st.markdown("### 📁 Data")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        st.caption(f"**{uploaded_file.name}** · {uploaded_file.size / 1024:.1f} KB")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">📊 Data Analyst Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload a dataset and ask questions about it in plain English.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Empty states
# ---------------------------------------------------------------------------
if not st.session_state.openai_key:
    st.info("👈 Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

if uploaded_file is None:
    st.info("👈 Upload a CSV or Excel file in the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Process file
# ---------------------------------------------------------------------------
temp_path, columns, df = preprocess_and_save(uploaded_file)

if not temp_path or columns is None or not isinstance(df, pd.DataFrame):
    st.error(f"Error processing file: {df if isinstance(df, str) else 'unknown error'}")
    st.stop()

# ---------------------------------------------------------------------------
# Quick stats
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
c4.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")

# ---------------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------------
with st.expander("📄 Preview data", expanded=False):
    st.dataframe(df, use_container_width=True)
    st.caption("Columns: " + ", ".join(columns))

st.divider()

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
data_analyst_agent = get_agent(st.session_state.openai_key, temp_path)

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
st.markdown("**Try asking:**")
example_queries = [
    "Summarize this dataset",
    "Are there any missing values?",
    "Show the top 5 rows by any numeric column",
    "Give me a breakdown by category",
]
cols = st.columns(len(example_queries))
for col, example in zip(cols, example_queries):
    if col.button(example):
        st.session_state.pending_query = example

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_query = st.chat_input("Ask a question about your data...")
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            try:
                response = data_analyst_agent.run(user_query)
                response_content = (
                    response.content if hasattr(response, "content") else str(response)
                )
                st.markdown(response_content)
                st.session_state.chat_history.append(("assistant", response_content))
            except Exception as e:
                error_msg = (
                    f"⚠️ Error generating response: {e}\n\n"
                    "Try rephrasing your query or check if the data format is correct."
                )
                st.error(error_msg)
                st.session_state.chat_history.append(("assistant", error_msg))
