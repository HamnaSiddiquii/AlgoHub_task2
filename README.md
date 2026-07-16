# Fieldnote — ReAct Agent with Web Search

**Week 2 · AlgoHub AI Agents & Automation Internship**

A ReAct (Reason + Act) agent that searches the live web, exposes its
Thought → Action → Observation loop in a real-time "Research Log," and
answers with inline numbered citations linked to the sources it actually
used.

## What it demonstrates

- **ReAct pattern** — the agent alternates between reasoning and tool
  calls instead of answering blind, and the loop is visible, not hidden.
- **Web search grounding** — every factual claim traces back to a live
  Tavily search result rather than the model's memory.
- **Answer grounding with citations** — the final answer cites sources
  as `[1]`, `[2]`, … with a matching source list underneath.

## Tech stack

| Layer | Choice |
|---|---|
| Agent framework | LangChain (`create_react_agent`) |
| Search tool | Tavily Search API |
| LLM | GPT-4o (OpenAI) or Claude (Anthropic) — switchable in the sidebar |
| UI | Streamlit, custom-styled |

## Project structure

```
week2_react_agent/
├── app.py              # Streamlit UI
├── agent.py            # ReAct prompt, agent executor, trace callback
├── requirements.txt
├── .env.example
└── .streamlit/
    └── config.toml     # base theme
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` and add your keys.

- An OpenAI or Anthropic API key, matching the provider you pick in the sidebar
- A [Tavily](https://tavily.com) API key (free tier is enough for this project)

## Run it

```bash
streamlit run app.py
```

Open the local URL Streamlit prints, pick a provider in the sidebar,
paste your keys, and ask a question that needs a current answer — e.g.
*"What's the latest stable version of LangChain?"* or *"Who won the
most recent F1 race?"*

## Dataset note

This project intentionally has no static dataset — the "data" is the
live web, retrieved fresh per question via the Tavily tool, per the
internship's dataset policy of not relying on pre-supplied data.

## Deploying
Live Demo: https://algoapptask2-n4cp8uzhc9quprtfqrhrqk.streamlit.app/
