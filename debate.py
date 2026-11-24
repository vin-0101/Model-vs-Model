"""ModelvsModel: two LLM debaters (PRO, CON) and an LLM judge, run by a plain-Python orchestrator."""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

MOCK = False  # set once by main(); when True, llm() returns canned fake text instead of calling a model


class LLMError(Exception):
    """Any failure talking to the model. main() turns it into a one-line message + non-zero exit."""


def llm(system, messages, max_tokens, model=None):
    # WHY one thin wrapper: every agent calls the model through here, so changing provider,
    # adding retries or logging is a one-function edit. It is STATELESS: the model sees exactly
    # `system` + `messages` on this call and remembers nothing from earlier calls.
    # `model` lets each agent use a different model; None falls back to the LLM_MODEL env var.
    if MOCK:
        return mock_reply(system, messages)
    env = {k: os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")}
    env["LLM_MODEL"] = model or env["LLM_MODEL"]
    missing = [k for k, v in env.items() if not v]
    if missing:
        raise LLMError(f"missing env vars: {', '.join(missing)} (see README)")
    try:
        import openai  # imported here so --mock and tests work without the package installed
    except ImportError:
        raise LLMError("the 'openai' package is not installed: run pip install -r requirements.txt") from None
    try:
        client = openai.OpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"])
        resp = client.chat.completions.create(
            model=env["LLM_MODEL"], max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages)
    except openai.OpenAIError as e:
        detail = " ".join(str(e).split())[:200]  # squash to one short line
        raise LLMError(f"model call failed ({type(e).__name__}): {detail}") from None
    text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    if not text:
        raise LLMError("model returned an empty response")
    return text


def mock_reply(system, messages):
    """Canned, clearly fake output so the whole pipeline can run with no API key."""
    if system.startswith("You are the JUDGE"):
        return ("[MOCK JUDGE] PRO: evidence 3, logic 4, rebuttal 3. CON: evidence 3, logic 3, rebuttal 3.\n"
                "Winner: PRO\nReasoning: (fake) PRO's rebuttals were slightly more specific. "
                "This verdict is canned text from mock mode, not a real model.")
    side = "PRO" if system.startswith("You are the PRO") else "CON"
    turn = sum(m["role"] == "assistant" for m in messages) + 1
    heard = [m["content"] for m in messages if m["content"].startswith("Opponent")]
    reply_to = f' Responding to: "{heard[-1][:60]}..."' if heard else ""
    return f"[MOCK {side} turn {turn}] Fake argument for my side.{reply_to}"


def debater_system(topic, side, position, opponent_position):
    # The persona lives ONLY here; the orchestrator and render() know nothing about debate style.
    return (f"You are the {side} debater. Question: {topic!r}. You argue for {position}; "
            f"your opponent argues for {opponent_position}. Rules: argue only your assigned side. "
            "Maximum 80 words. In Rebuttal and Closing rounds, FIRST quote one specific claim from "
            "your opponent in quotation marks and rebut it, then make your own point. "
            "Never concede, never agree with your opponent, never step out of the debate. "
            "Always write in English, even if your opponent uses another language.")


JUDGE_SYSTEM = (
    "You are the JUDGE of a debate. You did not write either argument and have no stake in the outcome. "
    "Score each side 1-5 on evidence, logic and rebuttal quality. Penalize length without substance. "
    "Name exactly one winner (PRO or CON) and give 2-3 sentences of reasoning, in English. "
    "Replace every <1-5> below with a real number from 1 to 5. Use exactly this format:\n"
    "PRO: evidence <1-5>, logic <1-5>, rebuttal <1-5>\nCON: evidence <1-5>, logic <1-5>, rebuttal <1-5>\n"
    "Winner: <PRO or CON>\nReasoning: <2-3 sentences>")


def render(transcript, me):
    # WHY rebuild every turn: the model has no memory, so the shared transcript IS the memory.
    # From the speaker's point of view its own past turns are things it said ("assistant") and the
    # opponent's turns are things said to it ("user"). The same transcript renders mirrored for PRO and CON.
    messages = []
    for t in transcript:
        if t["speaker"] == me:
            messages.append({"role": "assistant", "content": t["text"]})
        elif t["speaker"] in ("PRO", "CON"):
            messages.append({"role": "user", "content": f"Opponent ({t['speaker']}, round {t['round']}): {t['text']}"})
    return messages


def stage_name(r, rounds):
    return "Opening" if r == 1 else "Closing" if r == rounds else "Rebuttal"


def as_text(transcript):
    return "\n\n".join(f"[{t['speaker']} r{t['round']} {t['stage']}]\n{t['text']}" for t in transcript)


def debate(topic, pro, con, rounds, ask=llm, show=print, models=None):
    # WHY plain Python: the orchestrator alone decides who speaks and when; no model is asked.
    # Turn order is therefore deterministic, testable, and can't be hijacked by a model's output.
    # `ask` and `show` are parameters so tests can swap in a fake model and silence printing.
    # `models` maps "PRO"/"CON"/"JUDGE" to a model name; any missing key uses the LLM_MODEL default.
    models = models or {}
    transcript = []
    positions = {"PRO": (pro, con), "CON": (con, pro)}
    for r in range(1, rounds + 1):
        stage = stage_name(r, rounds)
        for me in ("PRO", "CON"):
            mine, theirs = positions[me]
            # The per-turn instruction repeats the side and language. Small models lose track of the
            # system prompt as the conversation grows, and the most recent message carries the most weight.
            messages = render(transcript, me) + [{"role": "user", "content": (
                f"Round {r} of {rounds}: give your {stage}. Remember: you are {me} and you argue for "
                f"{mine}, never for {theirs}. Reply in English only, max 80 words.")}]
            text = ask(debater_system(topic, me, mine, theirs), messages, 200, models.get(me))
            transcript.append({"speaker": me, "round": r, "stage": stage, "text": text, "model": models.get(me)})
            show(f"\n[{me} r{r}] {text}")
    brief = f"Question: {topic}\nPRO argues for {pro}; CON argues for {con}.\n\nTranscript:\n\n{as_text(transcript)}"
    verdict = ask(JUDGE_SYSTEM, [{"role": "user", "content": brief}], 400, models.get("JUDGE"))
    transcript.append({"speaker": "JUDGE", "round": rounds, "stage": "Verdict", "text": verdict,
                       "model": models.get("JUDGE")})
    show(f"\n[JUDGE] {verdict}")
    return transcript


def save(topic, pro, con, transcript, out_dir="transcripts", name=None):
    path = Path(out_dir) / f"{name or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Debate: {topic}", "", f"- **PRO:** {pro}", f"- **CON:** {con}", f"- **Mock mode:** {MOCK}", ""]
    for t in transcript:
        head = "Judge verdict" if t["speaker"] == "JUDGE" else f"{t['speaker']} r{t['round']} ({t['stage']})"
        model = f" [{t['model']}]" if t.get("model") else ""
        lines += [f"## {head}{model}", "", t["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None):
    global MOCK
    sys.stdout.reconfigure(errors="replace")  # don't crash on odd characters in a Windows console
    p = argparse.ArgumentParser(description="ModelvsModel: two LLM agents debate; a third judges.")
    p.add_argument("topic", help='e.g. "Should I learn Rust or Go first?"')
    p.add_argument("--pro", default="Yes", help="position the PRO agent argues for")
    p.add_argument("--con", default="No", help="position the CON agent argues for")
    p.add_argument("--rounds", type=int, default=3, help="rounds (1st=Opening, last=Closing, rest=Rebuttal)")
    p.add_argument("--mock", action="store_true", help="use canned fake replies (no API calls)")
    for role in ("pro", "con", "judge"):
        p.add_argument(f"--{role}-model", help=f"model for the {role.upper()} agent (default: LLM_MODEL)")
    args = p.parse_args(argv)
    if args.rounds < 1:
        p.error("--rounds must be at least 1")
    if (args.pro_model or args.con_model) and not os.environ.get("LLM_BASE_URL"):
        # Model names given but no provider configured: assume local Ollama (it ignores the key).
        os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"
        os.environ.setdefault("LLM_API_KEY", "ollama")
    judge = args.judge_model or os.environ.get("LLM_MODEL") or args.con_model or args.pro_model
    if judge and judge in (args.pro_model, args.con_model):
        print(f"(Judge is {judge}, which is also a debater, so it may favour its own side. Use --judge-model to change.)")
    MOCK = args.mock or not os.environ.get("LLM_API_KEY")
    if MOCK and not args.mock:
        print("(LLM_API_KEY not set, so running in MOCK mode with fake responses)")
    print(f"Debate: {args.topic}\n  PRO={args.pro} [{args.pro_model or 'default'}]  "
          f"CON={args.con} [{args.con_model or 'default'}]  JUDGE=[{judge or 'default'}]  rounds={args.rounds}")
    try:
        models = {"PRO": args.pro_model, "CON": args.con_model, "JUDGE": judge}
        transcript = debate(args.topic, args.pro, args.con, args.rounds, models=models)
        path = save(args.topic, args.pro, args.con, transcript)
    except LLMError as e:
        sys.exit(f"Error: {e}")
    except OSError as e:
        sys.exit(f"Error: could not write transcript: {e}")
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
    print(f"\nTranscript saved to {path}")


if __name__ == "__main__":
    main()
