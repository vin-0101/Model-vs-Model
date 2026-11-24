# ModelvsModel

Two local LLMs debate any question over multiple rounds while a third model judges the winner.

Ever wanted to watch two AI models argue about something they have no business
having opinions on? Pick a question, give each side a model, and let them go at it:
an opening statement, some rebuttals, and a closing argument. Then a third model
steps in as judge, scores both sides and declares a winner. Nobody concedes, and
there are no hard feelings, mostly because none of them have feelings.

## A taste of the action

**Question:** Which animal is better, a cat or a dog?
`qwen2.5:7b` argues for cats, `llama3.1` argues for dogs, and `llama2` judges.

> **[PRO r1, qwen2.5]** Cats are low-maintenance pets that require less space and attention
> than dogs. They are independent yet affectionate, perfect for those with busy lifestyles.
>
> **[CON r1, llama3.1]** Dogs bring people together, whereas cats often isolate us from one another.
>
> **[CON r3, llama3.1]** Unlike cats, dogs can be trained to assist people with disabilities
> and provide therapeutic benefits for those in need.
>
> **[JUDGE, llama2]** PRO: evidence 4, logic 4, rebuttal 4. CON: evidence 3, logic 3, rebuttal 3.
> **Winner: PRO.**

Team Cat takes it. The full transcripts are in [`transcripts/`](transcripts/).

The models have their quirks. Qwen sometimes drifts into Chinese mid-sentence,
then remembers it is supposed to be speaking English and carries on.

## Quick start (two local Ollama models)

You need [Ollama](https://ollama.com) running and a couple of models pulled
(`ollama pull qwen2.5:7b`, `ollama pull llama3.1`, and so on).

```
pip install -r requirements.txt
python debate.py "Which animal is better, a cat or a dog?" --pro cat --con dog --pro-model qwen2.5:7b --con-model llama3.1 --judge-model llama2
```

No env vars are needed: passing model names makes it use local Ollama automatically.
Each turn prints live as `[PRO r1]`, `[CON r1]` and so on, and the whole debate is saved
to `transcripts/<timestamp>.md`.

**Options**
- `--pro` / `--con`: the position each side argues for (default `Yes` / `No`).
- `--pro-model` / `--con-model` / `--judge-model`: which model plays each role. Run `ollama list`
  to see what you have. If you skip `--judge-model`, the CON model judges, which is a bit like
  letting a player referee their own match.
- `--rounds N`: how many rounds (default 3). Round 1 is the Opening, the last is the Closing,
  and everything in between is a Rebuttal.
- `--mock`: canned fake replies with no models called, handy for testing.

## Using a cloud provider instead

Any OpenAI-compatible endpoint works. Set three env vars (PowerShell shown; in bash, use
`export LLM_BASE_URL=...`):

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

**Ollama** (if you prefer env vars; the key can be any non-empty string)
```
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_API_KEY  = "ollama"
$env:LLM_MODEL    = "llama3.2"
```

Then run `python debate.py "Should I learn Rust or Go first?" --pro Rust --con Go`.
Model names change over time, so check your provider's current list if one is rejected.
Mock mode turns on automatically when `LLM_API_KEY` is not set and no model names are given.

## How it works

The models argue, but they don't run the show. The **orchestrator** (`debate()`) is
ordinary Python with no LLM in it. A `for` loop decides that PRO speaks, then CON,
round after round, and then the Judge speaks. No model ever chooses who talks next,
so the flow is deterministic and testable, and a model can't derail it.

Every **LLM call is stateless**: the model remembers nothing between calls. Before each
turn, `render(transcript, me)` rebuilds the whole conversation from one shared transcript,
as seen by that speaker. The speaker's own past turns become `assistant` messages
("what I said"), and the opponent's turns become `user` messages ("what was said to me").
The same transcript therefore renders mirror-imaged for PRO and CON.

Personas (debater rules, judge rubric) live only in system prompts, and every model call
goes through one thin wrapper, `llm(system, messages, max_tokens, model)`. Changing
provider, or adding retries or logging, means editing a single function.

No agent frameworks are used: it's plain Python plus the `openai` client library, which
also talks to Ollama. Run the tests with `python tests.py`.
