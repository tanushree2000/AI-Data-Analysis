"""
NeuroAnalyst // AI Data Console
--------------------------------
A fully self-contained, mockup AI Data Analytics Chatbot UI.

- NO real API keys, NO external LLM calls.
- Simulated streaming responses + a cached dummy pandas dataframe.
- "Cyberpunk Dark Dashboard" design system enforced via injected CSS.

Run with:  streamlit run cyberpunk_data_chatbot.py
"""

import time
import random

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# 🎨 COLOR PALETTE CONSTANTS  (kept in sync with the injected CSS below)
# =============================================================================
BG_DEEP = "#0B0E14"        # Global background
BG_PANEL = "#161A23"       # App / sidebar background
ACCENT_GREEN = "#00FFAA"   # Primary accent — metrics, primary borders
ACCENT_CYAN = "#00E5FF"    # Secondary accent — buttons, links, highlights
ACCENT_MAGENTA = "#FF3864" # Destructive / distinct-action accent
TEXT_BODY = "#E2E8F0"      # Body copy
TEXT_WHITE = "#FFFFFF"     # Bold headers
TEXT_MUTED = "#94A3B8"     # Subtext / captions
CARD_BG = "#1E293B"        # Card / container background
CARD_BORDER = "#334155"    # Card border

# =============================================================================
# ⚙️ PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="NeuroAnalyst // AI Data Console",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 💉 CUSTOM CSS INJECTION — Cyberpunk Dark Dashboard
# =============================================================================
st.markdown(
    """
    <style>
        /* ---------- Global background ---------- */
        [data-testid="stAppViewContainer"] {
            background-color: #0B0E14;
        }
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }
        [data-testid="stBottomBlockContainer"] {
            background-color: #0B0E14;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background-color: #161A23;
            border-right: 1px solid #334155;
        }
        [data-testid="stSidebar"] * {
            color: #E2E8F0;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #FFFFFF !important;
        }

        /* ---------- Text hierarchy ---------- */
        h1, h2, h3, h4 { color: #FFFFFF !important; font-weight: 700; }
        p, span, label { color: #E2E8F0; }
        [data-testid="stCaptionContainer"], .stCaption, small {
            color: #94A3B8 !important;
        }

        /* ---------- Generic cards ---------- */
        .cyber-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 18px 20px;
            height: 100%;
        }

        /* ---------- KPI cards ---------- */
        .kpi-card { text-align: left; }
        .kpi-icon {
            font-size: 1.4rem;
            color: #00E5FF;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 28px;
            font-weight: 800;
            color: #00FFAA;
            text-shadow: 0 0 14px rgba(0, 255, 170, 0.35);
            line-height: 1.2;
        }
        .kpi-label {
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #94A3B8;
            margin-top: 4px;
        }

        /* ---------- Buttons (secondary / default) ---------- */
        .stButton > button {
            background-color: #1E293B;
            color: #00E5FF;
            border: 1px solid #00E5FF;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.15s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #00E5FF;
            color: #0B0E14;
            box-shadow: 0 0 14px rgba(0, 229, 255, 0.55);
            border-color: #00E5FF;
        }

        /* ---------- "Clear Memory" primary / destructive button ---------- */
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
            background-color: #1E293B !important;
            color: #FF3864 !important;
            border: 2px solid #FF3864 !important;
            box-shadow: 0 0 10px rgba(255, 56, 100, 0.4);
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover,
        [data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {
            background-color: #FF3864 !important;
            color: #0B0E14 !important;
        }

        /* ---------- Inputs ---------- */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div,
        [data-testid="stFileUploaderDropzone"] {
            background-color: #1E293B !important;
            color: #E2E8F0 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }
        .stSlider [data-baseweb="slider"] > div > div {
            background: #334155 !important;
        }

        /* ---------- Expander ---------- */
        [data-testid="stExpander"] {
            background-color: #1E293B;
            border: 1px solid #334155 !important;
            border-radius: 12px;
        }
        [data-testid="stExpander"] summary {
            color: #FFFFFF !important;
            font-weight: 600;
        }

        /* ---------- Tabs ---------- */
        [data-testid="stTabs"] button[data-baseweb="tab"] {
            background-color: #1E222B;
            color: #94A3B8;
            border-radius: 8px 8px 0 0;
            border: 1px solid #334155;
            border-bottom: none;
            padding: 8px 18px;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #1E293B;
            color: #FFFFFF !important;
            border-bottom: 3px solid #00E5FF;
        }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #00E5FF !important;
        }

        /* ---------- Chat messages ---------- */
        [data-testid="stChatMessage"] {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 6px 4px;
        }

        /* ---------- Dataframes ---------- */
        [data-testid="stDataFrame"] {
            border: 1px solid #334155;
            border-radius: 10px;
        }

        /* ---------- Status rows (sidebar) ---------- */
        .status-row {
            font-size: 0.85rem;
            color: #94A3B8;
            padding: 3px 0;
        }
        .status-row b { color: #E2E8F0; }

        /* ---------- Scrollbar ---------- */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: #0B0E14; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #00E5FF; }

        hr { border-color: #334155 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 🧪 DUMMY DATA GENERATION  (cached — deterministic, no external source)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_dummy_data() -> pd.DataFrame:
    """Build a realistic, fully synthetic sales dataset."""
    rng = np.random.default_rng(42)
    n = 500

    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    products = [
        "Quantum SaaS", "NeuroAnalytics Pro", "DataStream API",
        "EdgeCompute Kit", "VisionAI Suite",
    ]
    dates = pd.date_range("2025-01-01", periods=180, freq="D")

    df = pd.DataFrame(
        {
            "Date": rng.choice(dates, n),
            "Region": rng.choice(regions, n),
            "Product": rng.choice(products, n),
            "Units_Sold": rng.integers(5, 500, n),
            "Customer_Satisfaction": np.round(rng.uniform(2.5, 5.0, n), 1),
        }
    )
    df["Revenue"] = np.round(df["Units_Sold"] * rng.uniform(20, 220, n), 2)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


df = load_dummy_data()


# =============================================================================
# 📈 DARK-THEME PLOTLY CHART BUILDER
# =============================================================================
def make_chart(source_df: pd.DataFrame):
    fig = px.scatter(
        source_df,
        x="Units_Sold",
        y="Revenue",
        color="Region",
        size="Customer_Satisfaction",
        hover_data=["Product"],
        title="📈 Revenue vs. Units Sold by Region",
        color_discrete_sequence=[ACCENT_GREEN, ACCENT_CYAN, ACCENT_MAGENTA, "#FFD166", "#A78BFA"],
    )
    fig.update_layout(
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_BODY, family="sans-serif"),
        title_font=dict(color=TEXT_WHITE, size=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_BODY)),
        margin=dict(l=10, r=10, t=55, b=10),
        height=420,
    )
    fig.update_xaxes(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER, color=TEXT_MUTED)
    fig.update_yaxes(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER, color=TEXT_MUTED)
    fig.update_traces(marker=dict(line=dict(width=1, color=BG_DEEP), opacity=0.85))
    return fig


# =============================================================================
# 🤖 SIMULATED RESPONSE ENGINE  (no external LLM — pure local logic)
# =============================================================================
GENERIC_REPLIES = [
    "Based on the {n}-row dataset currently loaded, regional performance looks fairly balanced — try asking me to **'plot revenue by region'** for a visual breakdown.",
    "I've cross-referenced the simulated dataset. Let me know if you'd like a **chart**, a **summary**, or a breakdown by **product**.",
    "That's an interesting question! Try asking me to **'summarize the dataset'**, **'show missing values'**, or **'plot a chart'** for deeper insight.",
]


def generate_dummy_response(query: str, source_df: pd.DataFrame):
    """Returns (response_type, payload) — response_type is 'chart' or 'text'."""
    q = query.lower()

    if any(k in q for k in ["chart", "plot", "graph", "visualiz"]):
        return "chart", None

    if any(k in q for k in ["missing", "null", "nan"]):
        missing = int(source_df.isna().sum().sum())
        verdict = "Nice — the data looks clean! ✅" if missing == 0 else "You may want to address these before deeper analysis."
        return "text", f"I scanned all **{len(source_df)} rows** across **{len(source_df.columns)} columns** and found **{missing} missing values**. {verdict}"

    if any(k in q for k in ["summar", "describe", "overview"]):
        return "text", (
            f"Here's a quick overview: the dataset spans **{source_df['Region'].nunique()} regions** "
            f"and **{source_df['Product'].nunique()} products**, with total revenue of "
            f"**${source_df['Revenue'].sum():,.0f}** and an average customer satisfaction score of "
            f"**{source_df['Customer_Satisfaction'].mean():.2f}/5**."
        )

    if any(k in q for k in ["top", "best", "highest"]):
        grouped = source_df.groupby("Product")["Revenue"].sum()
        top_product, top_rev = grouped.idxmax(), grouped.max()
        return "text", f"The top-performing product is **{top_product}**, generating **${top_rev:,.0f}** in total revenue."

    return "text", random.choice(GENERIC_REPLIES).format(n=len(source_df))


def stream_to_placeholder(placeholder, text: str, delay: float = 0.025) -> str:
    """Simulated streaming loop — types text into a placeholder word by word."""
    streamed = ""
    for word in text.split(" "):
        streamed += word + " "
        placeholder.markdown(streamed + "▌")
        time.sleep(delay)
    placeholder.markdown(streamed)
    return streamed


# =============================================================================
# 🕹️ SESSION STATE INIT
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to NeuroAnalyst.** I've loaded a simulated sales dataset with "
                f"**{len(df)} records**. Ask me to *'summarize the dataset'*, *'show missing values'*, "
                "or *'plot revenue by region'* to get started."
            ),
            "fig": None,
        }
    ]

# =============================================================================
# 🎛️ LEFT-HAND CONTROL PANEL (SIDEBAR)
# =============================================================================
with st.sidebar:
    st.markdown("## 🕹️ Control Panel")
    st.markdown("---")

    st.markdown("#### 🔑 Authentication")
    st.text_input(
        "API Key (Simulated)",
        type="password",
        placeholder="sk-••••••••••••••••",
        help="🔒 Demo mode — no real key required, nothing is sent anywhere.",
    )

    st.markdown("#### 📂 Dataset")
    uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.caption(f"✅ Received **{uploaded_file.name}** — this demo runs entirely on the built-in simulated dataset.")

    st.markdown("#### 🧠 Model Configuration")
    model_choice = st.selectbox(
        "Select Model",
        [
            "🟢 GPT-4o (Simulated)",
            "🔵 Claude 3.5 Sonnet (Simulated)",
            "🟣 Gemini 1.5 Pro (Simulated)",
            "🟠 Llama 3 70B (Simulated)",
        ],
    )

    st.markdown("#### 🎛️ Hyperparameters")
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)

    st.markdown("---")
    st.markdown(f"<div class='status-row'>🧠 Model: <b>{model_choice}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='status-row'>🎚️ Temperature: <b>{temperature}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='status-row'>💬 Messages: <b>{len(st.session_state.messages)}</b></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧹 Clear Memory", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

# =============================================================================
# 🏷️ HEADER
# =============================================================================
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom: 2px;">
        <span style="font-size:2.2rem;">🤖</span>
        <span style="font-size:2.1rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.02em;">
            NeuroAnalyst <span style="color:#00FFAA;">//</span> AI Data Console
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#94A3B8; margin-top:-4px; margin-bottom:22px;'>"
    "Autonomous data analysis workspace — simulated LLM backend, zero external API calls."
    "</p>",
    unsafe_allow_html=True,
)

# =============================================================================
# 📊 TOP DASHBOARD KPIs (4-COLUMN GRID)
# =============================================================================
def kpi_card(label: str, value: str, icon: str):
    st.markdown(
        f"""
        <div class="cyber-card kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Total Records", f"{len(df):,}", "🗂️")
with k2:
    kpi_card("Total Revenue", f"${df['Revenue'].sum():,.0f}", "💰")
with k3:
    kpi_card("Avg Satisfaction", f"{df['Customer_Satisfaction'].mean():.2f} / 5", "⭐")
with k4:
    kpi_card("Active Regions", f"{df['Region'].nunique()}", "🌐")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# 🔍 COLLAPSIBLE MULTI-TAB SCHEMA VIEW
# =============================================================================
with st.expander("🔍 Expand Structured Schema Inspection Matrix", expanded=False):
    tab1, tab2 = st.tabs(["🧬 Column Schema", "📄 Sample Records"])

    with tab1:
        schema_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Dtype": [str(t) for t in df.dtypes],
                "Non-Null Count": df.notna().sum().values,
                "Unique Values": [df[c].nunique() for c in df.columns],
            }
        )
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

st.markdown("---")

# =============================================================================
# 💬 INLINE CONVERSATIONAL CANVAS
# =============================================================================
st.markdown("### 💬 Conversational Analysis Feed")
st.caption(f"🧠 Active model: {model_choice}  ·  🎚️ Temperature: {temperature}")

# ---- Render chat history (static — cached figs prevent flicker/re-render) ----
for idx, msg in enumerate(st.session_state.messages):
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("fig") is not None:
            st.plotly_chart(msg["fig"], use_container_width=True, key=f"chart_history_{idx}")

# ---- Chat input + simulated streaming response ----
user_query = st.chat_input("Ask about your data... (try 'plot revenue by region')")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query, "fig": None})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        response_type, payload = generate_dummy_response(user_query, df)

        if response_type == "chart":
            intro_text = "📊 Generating a live visualization from your simulated dataset..."
            final_text = stream_to_placeholder(placeholder, intro_text, delay=0.035)

            fig = make_chart(df)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_new_{len(st.session_state.messages)}")

            st.session_state.messages.append({"role": "assistant", "content": final_text, "fig": fig})
        else:
            final_text = stream_to_placeholder(placeholder, payload, delay=0.02)
            st.session_state.messages.append({"role": "assistant", "content": final_text, "fig": None})

# =============================================================================
# 🦶 FOOTER
# =============================================================================
st.markdown(
    "<div style='margin-top:2rem; padding-top:1rem; border-top:1px solid #334155; "
    "text-align:center; color:#94A3B8; font-size:0.8rem;'>"
    "⚡ NeuroAnalyst Console — 100% simulated mockup · No API keys · No external calls"
    "</div>",
    unsafe_allow_html=True,
)
