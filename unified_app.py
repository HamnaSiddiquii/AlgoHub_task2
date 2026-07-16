# --- ADDED AT THE VERY TOP TO STOP THE WARNING FLOODS ---
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import re
import time
import socket
from typing import Callable, List, Tuple, Union

import streamlit as st
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from duckduckgo_search import DDGS
import dns.resolver

# Load variables from a local .env file
load_dotenv()

# ────────────────────────────────────────────────────────────────────────
# Force Clean Network Resolution Engine
# ────────────────────────────────────────────────────────────────────────
try:
    dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
    dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

    if not hasattr(socket, '_original_getaddrinfo'):
        socket._original_getaddrinfo = socket.getaddrinfo


    def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        try:
            if host and not host.replace('.', '').isdigit():
                answers = dns.resolver.resolve(host, 'A')
                if answers:
                    host = answers[0].to_text()
        except Exception:
            pass
        return socket._original_getaddrinfo(host, port, family, type, proto, flags)


    socket.getaddrinfo = custom_getaddrinfo
except Exception as e:
    pass

# ────────────────────────────────────────────────────────────────────────
# Core Agent Logic
# ────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Fieldnote, a careful research agent. You answer questions by
reasoning step by step and searching the live web whenever you need a fact
you are not certain of. You never invent sources.

Before each tool call, briefly state your reasoning as plain text.

The web_search tool returns numbered results like:
[1] Title
Snippet text
URL: https://example.com

When you cite a fact, use the EXACT number shown next to the result you are
drawing from (e.g. [1]) — do not invent new numbers and do not write your
own URL. Once you have enough grounded evidence, give a clear, well-organised
final answer with those inline numbered citations. Do not add a "Sources:"
section yourself — it will be generated automatically from the numbers you
used.

IMPORTANT — stopping rule: use at most 3 web_search calls in total, even if
results are weak or repetitive. If your first search returns unhelpful,
generic, or irrelevant results, immediately try ONE different, more specific
phrasing — do not repeat the same or a near-identical query. After 3 searches
total, STOP searching no matter what, and give your best final answer using
whatever grounded evidence you found, combined with your own general
knowledge where needed. Clearly note in the answer if live search results
were limited or unhelpful. Never call the tool a 4th time."""

PROVIDER_ENV_VARS = {
    "OpenAI (GPT-4o)": "OPENAI_API_KEY",
    "Google (Gemini)": "GOOGLE_API_KEY",
    "Anthropic (Claude)": "ANTHROPIC_API_KEY",
}

PROVIDER_DEFAULT_MODELS = {
    "OpenAI (GPT-4o)": "gpt-4o-mini",
    "Google (Gemini)": "gemini-flash-latest",
    "Anthropic (Claude)": "claude-3-5-haiku-latest",
}


class TraceEvent:
    def __init__(self, kind: str, title: str, body: str):
        self.kind = kind
        self.title = title
        self.body = body


class SourceRegistry:
    def __init__(self):
        self._entries: List[Tuple[str, str]] = []

    def reset(self):
        self._entries = []

    def add(self, url: str, title: str) -> int:
        for i, (u, _t) in enumerate(self._entries, start=1):
            if u == url:
                return i
        self._entries.append((url, title))
        return len(self._entries)

    def formatted_sources(self, used_numbers: List[int]) -> str:
        lines = []
        for n in used_numbers:
            if 1 <= n <= len(self._entries):
                url, _title = self._entries[n - 1]
                lines.append(f"[{n}] {url}")
        return "\n".join(lines)


def extract_text(content: Union[str, list, dict, None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    pieces.append(item["text"])
                elif "text" in item:
                    pieces.append(str(item["text"]))
        return "".join(pieces).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return ""
    return str(content)


def make_search_tool(registry: SourceRegistry, max_results: int = 5):
    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the live web via DuckDuckGo."""
        blocks = []
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            return f"Search is temporarily unavailable: {e}. Please wrap up your findings based on what you already know."

        if not raw_results:
            return "No web results found for this search query term."

        for r in raw_results:
            url = r.get("href") or r.get("link") or ""
            title = (r.get("title") or "").strip()
            snippet = (r.get("body") or "").strip()
            if not url:
                continue
            num = registry.add(url, title)
            blocks.append(f"[{num}] {title}\n{snippet}\nURL: {url}")

        return "\n\n".join(blocks) if blocks else "No usable results found."

    return web_search


def split_answer_and_sources(answer: str) -> Tuple[str, List[int]]:
    parts = re.split(r"\n?Sources:\s*\n?", answer, maxsplit=1)
    body = parts[0].strip()
    used_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", body)})
    return body, used_numbers


def build_llm(provider: str, api_key: str, model_name: str):
    if provider == "Google (Gemini)":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
            max_retries=3,
        )
    elif provider == "OpenAI (GPT-4o)":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.2)
    elif provider == "Anthropic (Claude)":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=0.2)
    raise ValueError(f"Unknown provider: {provider}")


def build_agent_executor(llm):
    registry = SourceRegistry()
    search_tool = make_search_tool(registry)
    agent = create_react_agent(llm, [search_tool], prompt=SYSTEM_PROMPT)
    return agent, registry


def run_agent(agent, registry: SourceRegistry, question: str, on_event: Callable[[TraceEvent], None]) -> Tuple[
    str, List[str]]:
    registry.reset()
    step = 0
    final_text = ""

    # ADDED HARD RECURSION LIMIT CONFIG TO STOP INFINITE LOOPS ENTIRELY
    stream_config = {"recursion_limit": 20}

    for chunk in agent.stream({"messages": [("user", question)]}, config=stream_config, stream_mode="values"):
        time.sleep(0.5)
        last = chunk["messages"][-1]
        msg_type = getattr(last, "type", None)

        if msg_type == "ai":
            tool_calls = getattr(last, "tool_calls", None)
            if tool_calls:
                step += 1
                thought_text = extract_text(last.content)
                if thought_text:
                    on_event(TraceEvent("thought", f"Thought {step}", thought_text))
                for tc in tool_calls:
                    query = tc.get("args", {}).get("query", "")
                    on_event(TraceEvent("action", f"Action {step} - web_search", f'Searching: "{query}"'))
            else:
                final_text = extract_text(last.content)
                on_event(TraceEvent("final", "Final reasoning", final_text))

        elif msg_type == "tool":
            text = extract_text(last.content).strip()
            if len(text) > 700:
                text = text[:700].rsplit(" ", 1)[0] + " ..."
            on_event(TraceEvent("observation", f"Observation {step}", text))

    body, used_numbers = split_answer_and_sources(final_text)
    sources_block = registry.formatted_sources(used_numbers)
    sources = [line.split(" ", 1)[1] for line in sources_block.splitlines()] if sources_block else []
    return body, sources


# ────────────────────────────────────────────────────────────────────────
# Streamlit Presentation Layout
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fieldnote Research Agent", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp, .mockup-header-box, .search-title-label, .trace-item-container, .findings-card, .source-pill-item {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background-color: #409c8c !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #75ceb3 !important;
        border-right: none !important;
    }

    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {
        color: #112d32 !important;
        font-weight: 600 !important;
    }

    .mockup-header-box {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        color: #112d32;
    }
    .mockup-header-box h1 {
        font-size: 3.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        margin: 0 !important;
        color: #112d32 !important;
    }
    .mockup-header-box h4 {
        font-size: 1.5rem !important;
        font-weight: 400 !important;
        margin: 0.2rem 0 0px 0 !important;
        color: #ffffff !important;
    }
    .mockup-header-box p {
        font-size: 1rem !important;
        color: #ffffff !important;
        opacity: 0.9;
        margin-top: 0.4rem !important;
    }

    .search-title-label {
        color: #112d32 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        color: #112d32 !important;
        font-size: 1.05rem !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    }

    .stButton > button {
        background-color: #112d32 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
        transition: background-color 0.2s !important;
        margin-top: 1rem !important;
    }
    .stButton > button:hover {
        background-color: #1a444b !important;
        color: #ffffff !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
    }

    .trace-item-container {
        background-color: rgba(17, 45, 50, 0.2);
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #112d32;
    }

    .findings-card {
        background-color: #ffffff !important;
        color: #112d32 !important;
        padding: 2rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    }
    .findings-card p, .findings-card li, .findings-card h1, .findings-card h2, .findings-card h3 {
        color: #112d32 !important;
    }

    .source-pill-item {
        background-color: rgba(17, 45, 50, 0.15);
        border: 1px solid rgba(17, 45, 50, 0.2);
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
    }
    .source-pill-item a {
        color: #ffffff !important;
        text-decoration: underline !important;
    }

    .error-card {
        background-color: #c94a4a !important;
        color: #ffffff !important;
        padding: 1.2rem !important;
        border-radius: 6px !important;
        margin-top: 1rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mockup-header-box">
        <h1>FIELD NOTE</h1>
        <h4>ReAct Research Agent</h4>
        <p>AlgoHub AI Agents & Automation Internship, Week 2 Project</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Configuration")
provider = st.sidebar.selectbox("LLM Provider", ["OpenAI (GPT-4o)", "Google (Gemini)", "Anthropic (Claude)"])

default_model = PROVIDER_DEFAULT_MODELS[provider]
env_var_name = PROVIDER_ENV_VARS[provider]
api_key = os.getenv(env_var_name, "")

model_name = st.sidebar.text_input("Model Name", value=default_model)

st.sidebar.markdown("---")
if api_key:
    st.sidebar.caption("API Credentials active (.env file synchronized)")
else:
    st.sidebar.caption(f"Target context variable {env_var_name} missing from environment parameters.")

st.markdown('<div class="search-title-label">Enter Your Search</div>', unsafe_allow_html=True)
question = st.text_input(
    label="Search Input Box",
    label_visibility="collapsed",
    placeholder="Your search here...",
)
run_clicked = st.button("Execute Research")

if run_clicked:
    if not api_key:
        st.error(
            f"Configuration fault: No target token detected for {provider}. Provide a valid {env_var_name} inside your local environment variable mappings.")
    elif not question.strip():
        st.warning("Validation failure: Target search field parameter cannot remain empty.")
    else:
        status_container = st.container()
        status_container = st.container()


        def streamlit_event_logger(event: TraceEvent):
            # Trace UI intentionally suppressed — agent still runs and
            # produces the final answer/sources, we just don't render steps.
            pass

        try:
            with st.spinner("Compiling processing automation components..."):
                llm = build_llm(provider, api_key, model_name)
                agent, registry = build_agent_executor(llm)

            with status_container:
                st.info(
                    "Execution sequence connected!")

            final_answer, verified_sources = run_agent(
                agent=agent,
                registry=registry,
                question=question,
                on_event=streamlit_event_logger,
            )

            status_container.empty()

            st.markdown("Synthesized Findings Summary")
            st.markdown(f'<div class="findings-card">', unsafe_allow_html=True)
            st.markdown(final_answer)
            st.markdown('</div>', unsafe_allow_html=True)

            if verified_sources:
                st.markdown("<br>Grounded References", unsafe_allow_html=True)
                for idx, src in enumerate(verified_sources, start=1):
                    st.markdown(
                        f"""<div class="source-pill-item">
                            <strong>[{idx}]</strong> <a href="{src}" target="_blank">{src}</a>
                        </div>""",
                        unsafe_allow_html=True
                    )

        except Exception as err:
            status_container.empty()
            err_msg = str(err)
            if "recursion_limit" in err_msg or "recursion" in err_msg.lower():
                st.markdown(
                    """<div class="error-card">
                        <strong>Agent Loop Stopped Safely</strong><br>
                        The ReAct agent hit its max processing steps without finalizing an answer due to unstable network connectivity with DuckDuckGo. 
                        Please try clicking "Execute Research" again to restart the request path.
                    </div>""",
                    unsafe_allow_html=True
                )
            elif "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                st.markdown(
                    """<div class="error-card">
                        <strong>Rate Limit Exhausted (API 429 Error)</strong><br>
                        The Gemini Free Tier key has run out of tokens for this minute. 
                        Please wait 30-60 seconds before executing another loop, or switch over to OpenAI/Anthropic in the sidebar panel.
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f'<div class="error-card"><b>Internal Breakdown Encountered:</b><br>{err_msg}</div>',
                            unsafe_allow_html=True)