#!/usr/bin/env python3
"""Accelerated issue #16 reproducer: large prefix-cache hits with short fresh suffixes.

usage: issue16_repro.py MINUTES [--sizes 200000,400000,800000,950000] [--out F.jsonl]

Builds one synthetic document per size, prefills each once (cold), then loops until MINUTES
elapse: pick a document, append a fresh 512-4096-token suffix, ask for a short answer. Every
other minute alternates between one stream and four concurrent streams. Each request records
the prompt and cached token counts; a request counts as a warm hit when at least half of its
prompt came from the cache. Stops at the first engine failure (HTTP error, empty answer or a
failed /health) and exits 1, so a crash is never averaged away. Exit 0 needs zero failures.
The plan this follows is in docs/issue-16-swa-crash.md (step 3).
"""
import argparse
import json
import random
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "deepseek-v4.1-flash"
WORDS = ("river stone lamp orbit cedar violet market engine harbor quiet copper meadow signal "
         "lantern granite window ladder thunder pepper saddle canyon velvet anchor marble").split()


def post(path, body, timeout):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def healthy():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=30) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def words(rnd, n):
    return " ".join(rnd.choice(WORDS) for _ in range(n))


def document(seed, tokens):
    # About 1.3 tokens per word for this vocabulary; calibrated by /tokenize below.
    rnd = random.Random(seed)
    text = f"Document {seed}.\n" + words(rnd, int(tokens / 1.3))
    n = len(post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False}, 600)["tokens"])
    return text + words(rnd, max(0, int((tokens - n) / 1.3)))


def ask(doc, suffix_tokens, rnd):
    content = f"{doc}\n\nNote {time.time_ns()}: {words(rnd, int(suffix_tokens / 1.3))}\nAnswer with one word: which word appears first in this note?"
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 16,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    t0 = time.time()
    d = post("/v1/chat/completions", body, 1800)
    u = d.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    text = (d["choices"][0]["message"].get("content") or "").strip()
    return {"s": round(time.time() - t0, 2), "prompt": u.get("prompt_tokens"), "cached": det.get("cached_tokens"),
            "suffix": suffix_tokens, "text": text[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("minutes", type=float)
    ap.add_argument("--sizes", default="200000,400000,800000,950000")
    ap.add_argument("--out", default="issue16_repro.jsonl")
    a = ap.parse_args()
    out = open(a.out, "a")
    lock = threading.Lock()
    stats = {"ok": 0, "warm": 0, "fail": 0}
    stop = threading.Event()

    def log(rec):
        with lock:
            out.write(json.dumps(rec) + "\n")
            out.flush()

    docs = []
    for i, size in enumerate(int(x) for x in a.sizes.split(",")):
        doc = document(1600 + i, size)
        r = ask(doc, 512, random.Random(i))
        log({"phase": "cold", "size": size, **r})
        print(f"cold {size}: prompt {r['prompt']} in {r['s']} s", flush=True)
        docs.append(doc)

    def worker(w, until):
        rnd = random.Random(w * 7919 + int(time.time()))
        while time.time() < until and not stop.is_set():
            doc = rnd.choice(docs)
            try:
                r = ask(doc, rnd.randint(512, 4096), rnd)
                ok = bool(r["text"])
            except Exception as exc:  # noqa: BLE001
                r, ok = {"err": repr(exc)[:200]}, False
            warm = ok and (r.get("cached") or 0) * 2 >= (r.get("prompt") or 1)
            with lock:
                stats["ok" if ok else "fail"] += 1
                stats["warm"] += warm
            log({"phase": "warm", "w": w, "ok": ok, "warm": warm, **r})
            if not ok:
                stop.set()

    end = time.time() + a.minutes * 60
    minute = 0
    while time.time() < end and not stop.is_set():
        conc = 1 if minute % 2 == 0 else 4
        until = min(end, time.time() + 60)
        ts = [threading.Thread(target=worker, args=(w, until)) for w in range(conc)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if not healthy():
            stats["fail"] += 1
            log({"phase": "health", "ok": False})
            stop.set()
        minute += 1
        print(json.dumps({"t": time.strftime("%H:%M:%S"), "conc": conc, **stats}), flush=True)
    print("ISSUE16 " + json.dumps(stats), flush=True)
    sys.exit(0 if stats["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
