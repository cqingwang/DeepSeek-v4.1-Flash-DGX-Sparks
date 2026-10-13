#!/usr/bin/env python3
"""Plain long-context probes against the DS TP4 endpoint (no harness magic).

cold <ktok>          one cold salted prompt of ~ktok*1000 tokens, max_tokens 16 -> TTFT, prefill tok/s
sessions <ktok>      A cold, A follow-up, B cold, A revisit -> cached_tokens per step (prefix eviction test)
needle <ktok>        passphrase at 50 % depth, greedy -> PASS/FAIL + TTFT
messages             /v1/messages: deepseek image token in text, thinking enabled without budget
Every result line is JSON; also appended to results.jsonl next to this file.
"""
import json, os, random, string, sys, time, urllib.request

BASE = os.environ.get("DS_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("DS_MODEL", "deepseek-v4.1-flash")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
WORDS = ("river stone lamp orbit cedar violet market engine harbor quiet copper meadow signal "
         "canvas thunder pocket garden silver marble lantern compass winter ember falcon").split()

def text(ktok, salt):
    rnd = random.Random(salt)
    n_words = int(ktok * 1000 * 0.75)  # ~0.75 words per token for plain English words
    out = [f"[doc {salt}]"]
    for i in range(n_words):
        out.append(rnd.choice(WORDS))
        if i % 24 == 23:
            out.append(f"({rnd.randrange(10**9)}).\n")
    return " ".join(out)

def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data, time.time() - t0

def chat(messages, max_tokens=16, **kw):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    body.update(kw)
    return post("/v1/chat/completions", body)

def emit(rec):
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rec["label"] = os.environ.get("LABEL", "")
    line = json.dumps(rec)
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")

def usage(d):
    u = d.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    return u.get("prompt_tokens"), det.get("cached_tokens", 0), u.get("completion_tokens")

def cold(ktok):
    salt = f"{time.time_ns()}-{random.randrange(10**9)}"
    d, dt = chat([{"role": "user", "content": text(ktok, salt) + "\n\nReply with the word OK."}])
    p, c, o = usage(d)
    emit({"probe": "cold", "ktok": ktok, "prompt_tokens": p, "cached": c, "seconds": round(dt, 2),
          "prefill_tok_s": round((p - (c or 0)) / dt, 1) if p else None})

def sessions(ktok):
    msgs = {}
    res = {}
    for name in ("A", "B"):
        msgs[name] = [{"role": "user", "content": text(ktok, f"{name}-{time.time_ns()}") + "\n\nSummarise in 5 words."}]
    def step(tag, name, follow):
        m = list(msgs[name])
        if follow:
            m += [{"role": "assistant", "content": res[name + "_ans"]},
                  {"role": "user", "content": "Now in 3 words."}]
        d, dt = chat(m)
        if not follow:
            res[name + "_ans"] = d["choices"][0]["message"]["content"] or "ok"
        p, c, _ = usage(d)
        res[tag] = {"prompt": p, "cached": c, "s": round(dt, 2)}
    step("A_cold", "A", False); step("A_follow", "A", True)
    step("B_cold", "B", False); step("A_revisit", "A", True)
    emit({"probe": "sessions", "ktok": ktok, **{k: v for k, v in res.items() if not k.endswith("_ans")}})

def needle(ktok):
    salt = f"{time.time_ns()}"
    pw = "".join(random.choice(string.ascii_uppercase) for _ in range(10))
    body = text(ktok, salt)
    mid = len(body) // 2
    body = body[:mid] + f"\nThe secret passphrase is {pw}.\n" + body[mid:]
    d, dt = chat([{"role": "user", "content": body + "\n\nWhat is the secret passphrase? Answer with it only."}], max_tokens=24)
    ans = d["choices"][0]["message"]["content"] or ""
    p, c, _ = usage(d)
    emit({"probe": "needle", "ktok": ktok, "prompt_tokens": p, "seconds": round(dt, 2), "pass": pw in ans, "answer": ans[:40]})

def messages():
    res = {}
    tok = "<｜deepseek_image｜>"
    for name, body in (
        ("image_token_text", {"model": MODEL, "max_tokens": 32,
                              "messages": [{"role": "user", "content": f"The log line was: {tok} done. Repeat the word done."}]}),
        ("thinking_enabled_no_budget", {"model": MODEL, "max_tokens": 256, "thinking": {"type": "enabled"},
                                        "messages": [{"role": "user", "content": "What is 2+2?"}]}),
    ):
        try:
            d, dt = post("/v1/messages", body, timeout=300)
            res[name] = {"ok": True, "s": round(dt, 2), "stop": d.get("stop_reason")}
        except urllib.error.HTTPError as e:
            res[name] = {"ok": False, "http": e.code, "body": e.read()[:200].decode(errors="replace")}
    emit({"probe": "messages", **res})

def overlap(ktok):
    import threading
    out = {}
    def long_req():
        salt = f"ov-{time.time_ns()}"
        d, dt = chat([{"role": "user", "content": text(ktok, salt) + "\n\nReply OK."}])
        out["long_s"] = round(dt, 2)
    t = threading.Thread(target=long_req); t.start()
    time.sleep(8)
    d, dt = chat([{"role": "user", "content": f"Say hello. {time.time_ns()}"}], max_tokens=8)
    out["short_s"] = round(dt, 2)
    t.join()
    emit({"probe": "overlap", "ktok": ktok, **out})

if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"cold": lambda: cold(int(args[0])), "sessions": lambda: sessions(int(args[0])),
     "needle": lambda: needle(int(args[0])), "messages": messages, "overlap": lambda: overlap(int(args[0]))}[cmd]()
