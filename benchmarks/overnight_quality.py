#!/usr/bin/env python3
"""Correctness gate for the overnight TP3 campaign (thinking off, greedy unless noted).

Checks: exact-answer tasks, JSON validity, generated code executed against tests,
the same tasks fired four at a time (concurrency), greedy repeatability, and
long-context needle retrieval at the requested sizes. Writes a JSON report whose
greedy outputs can be diffed against another configuration's.

Usage: BASE_URL=http://10.0.0.1:8888 python3 overnight_quality.py OUT.json [needle_k_tokens,...]
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://10.0.0.1:8888").rstrip("/")
MODEL = os.environ.get("SERVED_MODEL_NAME", "deepseek-v4.1-flash")


def chat(prompt, max_tokens=256, temperature=0.0, timeout=1800):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read())
    return obj["choices"][0]["message"].get("content") or "", obj.get("usage", {}), time.perf_counter() - t


def check_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    src = m.group(1) if m else text
    ns = {}
    try:
        exec(src, ns)  # model-written code, run locally on the benchmark host for the gate
        f = ns["is_prime"]
        cases = {0: False, 1: False, 2: True, 3: True, 4: False, 17: True, 21: False, 97: True, 7919: True, 7917: False}
        return all(bool(f(k)) == v for k, v in cases.items())
    except Exception:
        return False


def check_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        o = json.loads(m.group(0))
        return o.get("name") == "Ada" and o.get("age") == 36 and isinstance(o.get("skills"), list) and len(o["skills"]) == 3
    except Exception:
        return False


TASKS = [
    ("mul", "What is 17 * 23? Reply with only the number.", lambda t: t.strip().rstrip(".") == "391"),
    ("speed", "A train covers 120 km in 1.5 hours at constant speed. What is its speed in km/h? Reply with only the number.",
     lambda t: re.sub(r"[^0-9.]", "", t.strip()) in ("80", "80.0")),
    ("capital", "What is the capital city of Australia? Reply with one word.", lambda t: "canberra" in t.lower()),
    ("json", "Return only a JSON object with keys name (the string \"Ada\"), age (the integer 36) and skills "
             "(a list of exactly three strings). No prose, no code fence.", check_json),
    ("code", "Write a Python function is_prime(n: int) -> bool that returns whether n is prime (n may be 0 or 1). "
             "Reply with a single python code block and nothing else.", check_code),
    ("order", "Sort these words alphabetically and reply with them comma-separated, nothing else: pear, apple, mango, fig, banana",
     lambda t: re.sub(r"\s", "", t.lower()).rstrip(".") == "apple,banana,fig,mango,pear"),
    ("count", "How many times does the letter r appear in the word 'strawberry'? Reply with only the number.",
     lambda t: t.strip().rstrip(".") == "3"),
    ("sum", "Compute 1234 + 5678 - 999. Reply with only the number.", lambda t: t.strip().rstrip(".") == "5913"),
]

REPEAT_PROMPT = ("Write a detailed, flowing essay of about 800 words on the history and engineering "
                 "of lighthouses, from the Pharos of Alexandria to automated LED beacons. Use full "
                 "paragraphs only: no lists, no headings, no markdown.")


def needle(k_tokens, seed=7):
    rng = random.Random(seed + k_tokens)
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
    sentences = ["The archive room was quiet except for the hum of the ventilation system.",
                 "Researchers catalogued every folder by date, region and subject heading.",
                 "Rain streaked the tall windows while the afternoon light faded slowly.",
                 "A maintenance log recorded routine inspections of the heating pipes.",
                 "The committee postponed its decision until the next quarterly meeting."]
    n = int(k_tokens * 1000 / 13)  # ~13 tokens per sentence
    body = [rng.choice(sentences) for _ in range(n)]
    pos = n * 37 // 100
    body.insert(pos, f"Important: the vault access code is {code}. Remember it.")
    prompt = " ".join(body) + "\n\nWhat is the vault access code mentioned in the text above? Reply with only the code."
    t0 = time.perf_counter()
    try:
        text, usage, dt = chat(prompt, max_tokens=32)
        return {"k": k_tokens, "prompt_tokens": usage.get("prompt_tokens"), "ok": code in text,
                "answer": text[:80], "secs": dt, "error": None}
    except Exception as e:
        return {"k": k_tokens, "ok": False, "error": repr(e), "secs": time.perf_counter() - t0}


def main():
    out = sys.argv[1]
    needles = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else []
    rep = {"tasks": {}, "concurrent": {}, "repeat": {}, "needles": []}
    for name, prompt, ok in TASKS:
        text, usage, dt = chat(prompt)
        rep["tasks"][name] = {"ok": bool(ok(text)), "text": text[:400],
                              "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(chat, p): (n, ok) for n, p, ok in TASKS}
        for f in concurrent.futures.as_completed(futs):
            n, ok = futs[f]
            try:
                text = f.result()[0]
                rep["concurrent"][n] = {"ok": bool(ok(text)),
                                        "same_as_c1": hashlib.sha256(text.encode()).hexdigest()[:16] == rep["tasks"][n]["sha"]}
            except Exception as e:
                rep["concurrent"][n] = {"ok": False, "error": repr(e)}
    shas = [hashlib.sha256(chat(REPEAT_PROMPT, max_tokens=256)[0].encode()).hexdigest()[:16] for _ in range(3)]
    rep["repeat"] = {"shas": shas, "distinct": len(set(shas))}
    for k in needles:
        rep["needles"].append(needle(k))
    n_ok = sum(v["ok"] for v in rep["tasks"].values())
    c_ok = sum(v.get("ok", False) for v in rep["concurrent"].values())
    rep["summary"] = {"tasks_ok": f"{n_ok}/{len(TASKS)}", "concurrent_ok": f"{c_ok}/{len(TASKS)}",
                      "concurrent_same_as_c1": sum(v.get("same_as_c1", False) for v in rep["concurrent"].values()),
                      "greedy_repeat_distinct": rep["repeat"]["distinct"],
                      "needles": [(n["k"], n.get("prompt_tokens"), n["ok"], round(n["secs"], 1)) for n in rep["needles"]]}
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep["summary"]))
    for n, v in rep["tasks"].items():
        if not v["ok"]:
            print("FAIL", n, repr(v["text"][:200]))


if __name__ == "__main__":
    main()
