#!/usr/bin/env python3
"""soak.py MINUTES CONC : CONC workers loop mixed chat requests for MINUTES; count errors, empty and non-text outputs.
Every ~20th request carries a 30-60K-token cold prompt so chunked prefill and the small-request bypass run under load."""
import json, os, random, sys, threading, time, urllib.request
BASE = os.environ.get("DS_URL", "http://127.0.0.1:8888"); MODEL = "deepseek-v4.1-flash"
mins, conc = float(sys.argv[1]), int(sys.argv[2]); stop = time.time() + mins * 60
P = ["Write a Python function that merges overlapping intervals, with tests.",
     "Describe a harbour town at dawn in 150 words.",
     'Return JSON {"name": str, "skills": [3 strings], "years": int} for a fictional engineer. JSON only.',
     "Explain TCP congestion control to a student in 200 words.",
     "Count from 1 to 120 separated by commas."]
WORDS = "river stone lamp orbit cedar violet market engine harbor quiet copper meadow".split()
stats = {"ok": 0, "err": 0, "empty": 0, "long_ok": 0, "lat": []}; lock = threading.Lock()
def one(i):
    long_req = random.random() < 0.05
    content = random.choice(P)
    if long_req:
        n = random.randint(22000, 45000)
        content = " ".join(random.choice(WORDS) for _ in range(n)) + f" [{time.time_ns()}]\nSummarise in one sentence."
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": random.choice([256, 512, 1024]),
            "temperature": random.choice([0, 0.6, 1.0]), "chat_template_kwargs": {"thinking": random.random() < 0.3}}
    t0 = time.time()
    try:
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=900))
        m = d["choices"][0]["message"]; text = (m.get("content") or "") + (m.get("reasoning_content") or "")
        with lock:
            if text.strip(): stats["ok"] += 1; stats["long_ok"] += long_req
            else: stats["empty"] += 1
            stats["lat"].append(time.time() - t0)
    except Exception as e:
        with lock: stats["err"] += 1; print("ERR", type(e).__name__, str(e)[:150], flush=True)
def worker(w):
    while time.time() < stop: one(w)
ts = [threading.Thread(target=worker, args=(w,)) for w in range(conc)]
[t.start() for t in ts]
while any(t.is_alive() for t in ts):
    time.sleep(60)
    with lock: print(json.dumps({"t": time.strftime("%H:%M:%S"), "ok": stats["ok"], "err": stats["err"], "empty": stats["empty"], "long_ok": stats["long_ok"]}), flush=True)
lat = sorted(stats["lat"]) or [0]
print(json.dumps({"final": True, "ok": stats["ok"], "err": stats["err"], "empty": stats["empty"], "long_ok": stats["long_ok"], "p50_s": round(lat[len(lat)//2], 1), "p99_s": round(lat[int(len(lat)*0.99)], 1)}), flush=True)
