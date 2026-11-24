"""Plain-assert tests. Run: python tests.py"""
import os
import tempfile
from pathlib import Path

import debate

T = [  # a tiny hand-made transcript: PRO r1, CON r1, PRO r2, then a judge entry
    {"speaker": "PRO", "round": 1, "stage": "Opening", "text": "p1"},
    {"speaker": "CON", "round": 1, "stage": "Opening", "text": "c1"},
    {"speaker": "PRO", "round": 2, "stage": "Closing", "text": "p2"},
    {"speaker": "JUDGE", "round": 2, "stage": "Verdict", "text": "j"},
]


def test_render_roles():
    pro = debate.render(T, "PRO")
    con = debate.render(T, "CON")
    assert [m["role"] for m in pro] == ["assistant", "user", "assistant"], pro
    assert [m["role"] for m in con] == ["user", "assistant", "user"], con
    assert pro[0]["content"] == "p1" and "c1" in pro[1]["content"]  # own words verbatim, opponent's labeled
    assert con[1]["content"] == "c1" and "p1" in con[0]["content"]
    assert all("j" != m["content"] for m in pro + con)  # judge never leaks into debater prompts


def test_turn_count_and_order():
    calls = []

    def fake_ask(system, messages, max_tokens, model=None):
        calls.append((system, model))
        return f"reply {len(calls)}"

    models = {"PRO": "model-a", "CON": "model-b", "JUDGE": "model-j"}
    t = debate.debate("Topic?", "A", "B", 3, ask=fake_ask, show=lambda *_: None, models=models)
    assert len(calls) == 3 * 2 + 1, len(calls)
    assert [x["speaker"] for x in t] == ["PRO", "CON"] * 3 + ["JUDGE"]
    assert [x["stage"] for x in t[::2]] == ["Opening", "Rebuttal", "Closing", "Verdict"]
    assert calls[-1][0].startswith("You are the JUDGE")
    assert [m for _, m in calls] == ["model-a", "model-b"] * 3 + ["model-j"]  # each agent got its own model


def test_transcript_file_written():
    with tempfile.TemporaryDirectory() as d:
        path = debate.save("Topic?", "A", "B", T, out_dir=d)
        assert path.exists() and path.parent == Path(d)
        body = path.read_text(encoding="utf-8")
        assert "# Debate: Topic?" in body and "## Judge verdict" in body and "p2" in body


def test_mock_pipeline_end_to_end():
    debate.MOCK = True
    try:
        t = debate.debate("Topic?", "A", "B", 2, show=lambda *_: None)  # uses the real llm() in mock mode
    finally:
        debate.MOCK = False
    assert len(t) == 5 and t[-1]["text"].startswith("[MOCK JUDGE]")
    assert 'Responding to: "Opponent (PRO' in t[1]["text"]  # CON actually saw PRO's opening


def test_missing_env_raises_clean_error():
    saved = {k: os.environ.pop(k, None) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")}
    try:
        debate.llm("sys", [{"role": "user", "content": "hi"}], 10)
        assert False, "expected LLMError"
    except debate.LLMError as e:
        assert "missing env vars" in str(e)
    finally:
        os.environ.update({k: v for k, v in saved.items() if v})


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print(f"PASS {f.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
