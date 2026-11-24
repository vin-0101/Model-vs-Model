# ModelvsModel

Two local LLMs debate any question over multiple rounds while a third model judges the winner.

Two LLM agents (PRO and CON) argue opposite sides of a question over N rounds
(Opening, Rebuttal..., Closing). A third Judge agent reads the full transcript,
scores both sides and picks a winner. Each turn prints live, and the whole debate
is saved to `transcripts/<timestamp>.md`.

## Setup

```
pip install -r requirements.txt
```

Set three env vars for any OpenAI-compatible endpoint. The examples use PowerShell;
in bash, use `export LLM_BASE_URL=...` instead.

**Groq** (key from console.groq.com)
```
$env:LLM_BASE_URL = "https://api.groq.com/openai/v1"
$env:LLM_API_KEY  = "gsk_..."
$env:LLM_MODEL    = "llama-3.3-70b-versatile"
```

**Gemini** (key from aistudio.google.com)
```
$env:LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
$env:LLM_API_KEY  = "AIza..."
$env:LLM_MODEL    = "gemini-2.5-flash"
```

**Ollama** (local, run `ollama pull llama3.2` first; the key can be any non-empty string)
```
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_API_KEY  = "ollama"
$env:LLM_MODEL    = "llama3.2"
```

Model names change over time, so check your provider's current model list if one is rejected.

## Run

```
python debate.py "Should I learn Rust or Go first?" --pro Rust --con Go --rounds 3
```

Add `--mock` to use canned fake replies with no API calls. Mock mode also turns on
automatically when `LLM_API_KEY` is not set. Run the tests with `python tests.py`.

## Two Ollama models debating

With Ollama running, give each side a model. No env vars are needed: passing model
names makes it use local Ollama automatically.

```
python debate.py "Should I learn Rust or Go first?" --pro Rust --con Go --pro-model qwen2.5:7b --con-model llama3.1
```

To try a different matchup, just change the model names (`ollama list` shows what you have).
The judge defaults to the CON model; pick a neutral one with `--judge-model llama2`.

## How it works

The **orchestrator** (`debate()`) is ordinary Python with no LLM in it. A `for` loop
decides that PRO speaks, then CON, round after round, and then the Judge speaks.
Agents only produce text. They never choose who speaks next, so the flow is
deterministic and testable, and a model can't derail it. Every **LLM call is stateless**:
the model remembers nothing between calls, so before each turn `render(transcript, me)`
rebuilds the whole conversation from the one shared transcript list, as seen by that speaker.
The speaker's own past turns become `assistant` messages ("what I said"), and the
opponent's turns become `user` messages ("what was said to me"). The same transcript
therefore renders mirror-imaged for PRO and CON. Personas (debater rules, judge rubric)
live only in system prompts. Every model call goes through one thin wrapper,
`llm(system, messages, max_tokens)`, so changing provider or adding retries or logging
means editing a single function.
