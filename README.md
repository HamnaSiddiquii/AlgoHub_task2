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

Copy `.env.example` to `.env` and add your keys, **or** just paste them
into the sidebar when the app is running (nothing is written to disk
either way):

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

Push this folder to a GitHub repo, then deploy for free on
[Streamlit Community Cloud](https://share.streamlit.io) pointing at
`app.py`. Add `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`
as app secrets rather than committing them.

## Submission checklist (per handbook §2)

- [ ] Push code to `AlgoHub_Task2_ReActAgent` repo with this README,
      screenshots, and results
- [ ] Zip as `AlgoHub_Task2_ReActAgent.zip`
- [ ] Record a 5–10 min demo: overview → a live query → the Research
      Log lighting up → the cited final answer
- [ ] Deploy (Streamlit Cloud link) and add the live URL here
- [ ] Post on LinkedIn tagging AlgoHub Software House, with the GitHub
      link, demo video, and a screenshot
