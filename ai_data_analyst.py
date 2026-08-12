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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ---- Global / font ---- */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }

        /* ---- Centered chat column (ChatGPT/Claude/Gemini-style) ---- */
        .block-container {
            max-width: 48rem;
            padding-top: 2.2rem;
            padding-bottom: 7rem;
            margin: 0 auto;
        }

        /* ---- Top bar ---- */
        .app-topbar {
            font-size: 1.05rem;
            font-weight: 600;
            color: #202123;
            margin-bottom: 1.2rem;
        }

        /* ---- Empty / welcome state ---- */
        .empty-state {
            text-align: center;
            padding: 3rem 0 1.2rem 0;
        }
        .empty-title {
            font-size: 1.6rem;
            font-weight: 600;
            color: #202123;
            margin-bottom: 0.3rem;
        }
        .empty-sub {
            font-size: 0.95rem;
            color: #8e8ea0;
            margin-bottom: 1.3rem;
        }

        /* ---- Buttons (chips + sidebar) ---- */
        .stButton>button {
            border-radius: 999px;
            border: 1px solid #e5e5e5;
            background-color: #ffffff;
            color: #353740;
            font-size: 0.85rem;
            font-weight: 500;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #f7f7f8;
            border-color: #d9d9d9;
            color: #202123;
        }
        section[data-testid="stSidebar"] .stButton>button {
            width: 100%;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background-color: #f7f7f8;
            border-right: 1px solid #ececec;
        }
        .sidebar-section-title {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #8e8ea0;
            margin-top: 0.5rem;
            margin-bottom: 0.4rem;
        }
        .status-row {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            padding: 3px 0;
            color: #40414f;
        }

        /* ---- Chat messages ---- */
        div[data-testid="stChatMessage"] {
            background-color: transparent;
            padding: 0.7rem 0;
            border-bottom: 1px solid #f0f0f0;
        }

        /* ---- Chat input ---- */
        div[data-testid="stChatInput"] {
            border: 1px solid #e5e5e5;
            border-radius: 1.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        }

        /* ---- Expander ---- */
        div[data-testid="stExpander"] {
            border: 1px solid #ececec;
            border-radius: 10px;
        }

        /* ---- Footer ---- */
        .app-footer {
            margin-top: 2rem;
            padding-top: 0.8rem;
            color: #b4b4c0;
            font-size: 0.75rem;
            text-align: center;
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
    st.markdown('<p class="sidebar-section-title">⚙️ Setup</p>', unsafe_allow_html=True)
    openai_key = st.text_input(
        "OpenAI API key", type="password", value=st.session_state.openai_key,
        help="Your key is kept only in this session and never stored.",
    )
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("API key saved", icon="✅")
    else:
        st.warning("Enter your OpenAI API key to get started.")

    st.divider()
    st.markdown('<p class="sidebar-section-title">📁 Data</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        st.caption(f"**{uploaded_file.name}** · {uploaded_file.size / 1024:.1f} KB")

    st.divider()
    st.markdown('<p class="sidebar-section-title">Status</p>', unsafe_allow_html=True)
    key_icon = "✅" if st.session_state.openai_key else "⬜"
    file_icon = "✅" if uploaded_file is not None else "⬜"
    st.markdown(
        f'<div class="status-row">{key_icon} API key connected</div>'
        f'<div class="status-row">{file_icon} Dataset uploaded</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="app-topbar">📊&nbsp; Data Analyst</div>', unsafe_allow_html=True)

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
st.caption(
    f"📄 {uploaded_file.name} · {len(df):,} rows · {len(df.columns)} columns · "
    f"{int(df.isna().sum().sum()):,} missing values · {uploaded_file.size / 1024:.1f} KB"
)

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
# Example queries — shown as a welcome screen before the first message
# ---------------------------------------------------------------------------
example_queries = [
    "Summarize this dataset",
    "Are there any missing values?",
    "Show the top 5 rows by any numeric column",
    "Give me a breakdown by category",
]

if not st.session_state.chat_history:
    st.markdown('<div class="empty-state">', unsafe_allow_html=True)
    st.markdown('<p class="empty-title">💬 Ask me anything about your data</p>', unsafe_allow_html=True)
    st.markdown('<p class="empty-sub">Try one of these to get started</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    cols = st.columns(len(example_queries))
    for col, example in zip(cols, example_queries):
        if col.button(example):
            st.session_state.pending_query = example

CHAT_AVATARS = {"user": "🧑", "assistant": "📊"}

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for role, content in st.session_state.chat_history:
    with st.chat_message(role, avatar=CHAT_AVATARS.get(role)):
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
    with st.chat_message("user", avatar=CHAT_AVATARS["user"]):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=CHAT_AVATARS["assistant"]):
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

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-footer">Powered by GPT-4o · DuckDB · Streamlit</div>',
    unsafe_allow_html=True,
)
