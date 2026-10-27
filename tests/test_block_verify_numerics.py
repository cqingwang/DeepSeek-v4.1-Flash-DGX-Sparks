"""Deterministic FP32 arithmetic checks for block_verify.block_accept_probs against an FP64 oracle.

test_block_verify.py checks losslessness statistically on a 5-token toy; a chi-square over 400k
samples cannot see a threshold error that only exists when probabilities sit within a few ULPs
of 1. This test drives the thresholds h_i directly: for each position i, every other position's
draw is +inf (never accepted), so tau is i + 1 exactly when eta_i <= h_i. A draw just below and
just above the FP64 threshold, computed on the same FP32 probabilities, must be accepted and
rejected. The tolerance is the FP32 rounding bound of the formula (see _tol).

Cases:
  1. the reassociation counterexamples: w = 1 - 2^-24 with S = 3e-8 (h 0.335, the old
     (S + 1) - w gave 0.503), and a clipped w = 1 with S = 1.4 * 2^-24 (h 1, the old gave 0.70);
  2. a grid of 1 - w in {0, k * 2^-24, 1e-6 .. 0.5} x S in [1e-10, 1e-3], at positions 1 and 3;
  3. production-shaped rows: V = 129280 bf16 logits with top-token margins 2..30, target at T = 1,
     draft at T * 0.7, probabilities = the FP64 softmax rounded to FP32 and clamped at 1;
  4. ties, -inf (top-k filtered target, zero draft mass), one-hot drafts, p == q (S = 0 at w = 1),
     truncated live blocks, bf16 inputs, an all-NaN row;
  5. on ordinary rows the threshold is exactly s / (s + (1 - w)) and moves by no more than the old
     association's rounding; the random draws consumed per call are unchanged.

Credit: the cancellation class follows Barron, Gnanapragasam and Tschannen, "A Fix for Softmax
and Cross-Entropy" (https://jonbarron.info/softmaxfix/), as Christopher Owen applied it to
verification kernels in spark-ds41f #584.
"""
import math
import os

import torch

import block_verify as bv

torch.manual_seed(0)
DEV = torch.device(os.environ.get("BV_TEST_DEVICE", "cpu"))   # cuda: the same checks on the GPU's arithmetic
U = 2.0 ** -24          # FP32 unit roundoff (half an ULP of 1)
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def w_fp32(X, P, Q, live=None):
    """w_i exactly as block_verify forms it (FP32 ratio, FP32 running product, clip at 1)."""
    bs, g = X.shape
    p_tok = P[:, :g].gather(-1, X.unsqueeze(-1)).squeeze(-1).float()
    q_tok = Q.gather(-1, X.unsqueeze(-1)).squeeze(-1).float()
    ratio = p_tok / q_tok.clamp_min(1e-30)
    if live is not None:
        alive = torch.arange(g).view(1, -1) < live.view(-1, 1)
        ratio = torch.where(alive, ratio, torch.zeros_like(ratio))
    cur, ws = torch.ones(bs, device=X.device), []
    for i in range(g):
        cur = torch.minimum(cur * ratio[:, i], torch.ones_like(cur))
        ws.append(cur)
    return torch.stack(ws, 1)


def ref(X, P, Q, live=None):
    """FP64 S_i and h_i on the given FP32 probabilities and on w_i as block_verify forms it.

    w is an FP32 quantity by construction (a ratio of FP32 probabilities, so 1 - w is a multiple
    of 2^-24 below 1); its rounding is representation error that only a log-domain pipeline can
    remove, and it is not what this test is about. Everything after w is FP64 here.

    Reference contract: FP32 inputs, FP32 w, FP64 S and h. This isolates the patched expression; it is
    NOT FP64 block verification from logits (nor from FP32 inputs with an FP64 ratio recursion)."""
    bs, g = X.shape
    w = w_fp32(X, P, Q, live).double()
    P, Q = P.double(), Q.double()
    if live is not None:
        alive = torch.arange(g).view(1, -1) < live.view(-1, 1)
        Q = Q * alive.unsqueeze(-1).double()
    h = torch.full((bs, g), math.nan, dtype=torch.float64)
    s = (w[:, :g - 1, None] * P[:, 1:g] - Q[:, 1:g]).clamp_min(0).sum(-1)
    d = 1.0 - w[:, :g - 1]
    den = s + d
    h[:, :g - 1] = torch.where(den > 0, s / den.clamp_min(1e-300), h[:, :g - 1])
    h[:, g - 1] = w[:, g - 1]
    # FP32 error of S: the product w * p rounds (U relative) wherever the term can be positive, and
    # the clamp can then admit a term that is negative by at most that rounding; the V-wide sum
    # adds a relative error that stays below 64 U for torch's blocked reductions at V <= 2^17.
    near = (w[:, :g - 1, None] * P[:, 1:g]) >= Q[:, 1:g] * (1 - 4 * U)
    err_s = 2 * U * (w[:, :g - 1, None] * P[:, 1:g] * near).sum(-1) + 64 * U * s
    return w, s, d, h, err_s


def _tol(h, s, d, err_s):
    """Bound on |h_fp32 - h_fp64| when (1 - w) is formed first: S's own error through
    dh/dS = d / (S + d)^2, plus a few roundings of the sum and the division."""
    t = 8 * U * h.abs() + 1e-30
    t[:, :-1] += err_s * d / (s + d).clamp_min(1e-300) ** 2
    return t


def run_bv(*args, **kw):
    """block_accept_probs on DEV (inputs built on the CPU; outputs back on the CPU)."""
    mv = lambda t: t.to(DEV) if isinstance(t, torch.Tensor) else t
    tau, corr = bv.block_accept_probs(*[mv(a) for a in args], **{k: mv(v) for k, v in kw.items()})
    return tau.cpu(), corr.cpu()


def decide(X, P, Q, i, eta_i, live=None):
    bs, g = X.shape
    eta = torch.full((bs, g), math.inf)
    eta[:, i] = eta_i.float()
    gumbel = torch.ones(bs, P.shape[-1])
    tau, _ = run_bv(X, P, Q, eta=eta, gumbel=gumbel, live=live)
    return tau == i + 1


def straddle(name, X, P, Q, live=None):
    """Draws at h_ref -/+ tol must be accepted / rejected at every position with a finite h_ref."""
    w, s, d, h, err_s = ref(X, P, Q, live)
    tol = _tol(h, s, d, err_s)
    bs, g = X.shape
    bad = 0
    for i in range(g):
        hi, ti = h[:, i], tol[:, i]
        ok = torch.isfinite(hi)
        if live is not None:
            ok &= i < live
        lo = (hi - ti).clamp(min=0)
        up = hi + ti
        acc_lo = decide(X, P, Q, i, lo, live)
        acc_up = decide(X, P, Q, i, up, live)
        must_acc = ok & (hi - ti > 0)                     # a draw at 0 says nothing when h ~ 0
        must_rej = ok & (up < 1)                          # draws are < 1: h >= 1 always accepts
        for r in torch.nonzero(must_acc & ~acc_lo).flatten().tolist():
            bad += 1
            if bad <= 6:
                FAILS.append(f"{name}: row {r} pos {i + 1}: draw {float(lo[r]):.10g} rejected, FP64 h = {float(hi[r]):.10g} "
                             f"(w = 1 - {float(d[r, min(i, g - 2)]):.3e}, S = {float(s[r, min(i, g - 2)]):.3e})")
        for r in torch.nonzero(must_rej & acc_up).flatten().tolist():
            bad += 1
            if bad <= 6:
                FAILS.append(f"{name}: row {r} pos {i + 1}: draw {float(up[r]):.10g} accepted, FP64 h = {float(hi[r]):.10g} "
                             f"(w = 1 - {float(d[r, min(i, g - 2)]):.3e}, S = {float(s[r, min(i, g - 2)]):.3e})")
    if bad > 6:
        FAILS.append(f"{name}: ... {bad} straddle failures in all")
    return bad


def two_pos_rows(pairs, V=4, g=2, at=0):
    """Rows where position `at` has weight w (the drafted token 0 has p = w, q = 1, all earlier
    positions p = q = one-hot) and the next position has S = w * sigma (q one-hot on token 0,
    p = [1 - sigma, sigma, 0, ...]). Every later position is p = q = one-hot on token 0."""
    n = len(pairs)
    P = torch.zeros(n, g + 1, V)
    Q = torch.zeros(n, g, V)
    X = torch.zeros(n, g, dtype=torch.long)
    for r, (w, sigma) in enumerate(pairs):
        for j in range(g + 1):
            P[r, j, 0] = 1.0
        for j in range(g):
            Q[r, j, 0] = 1.0
        P[r, at, 0], P[r, at, 1] = w, 1.0 - w
        P[r, at + 1, 0], P[r, at + 1, 1] = 1.0 - sigma, sigma
    return X, P, Q


def test_counterexamples():
    w = 1.0 - U
    X, P, Q = two_pos_rows([(w, 3e-8), (1.0, 1.4 * U)])
    _, s, d, h, _ = ref(X, P, Q)
    check(abs(float(h[0, 0]) - 0.33480405) < 1e-6, f"oracle h {float(h[0, 0])}")
    check(float(h[1, 0]) == 1.0, f"oracle h {float(h[1, 0])}")
    # 0.40 lies between the true 0.335 and the old association's 0.503; 0.80 between 0.70 and 1
    acc = decide(X, P, Q, 0, torch.tensor([0.40, 0.80]))
    check(not bool(acc[0]), "w = 1 - 2^-24, S = 3e-8: a draw of 0.40 accepted, true h = 0.335")
    check(bool(acc[1]), "w = 1 (clipped), S = 1.4 * 2^-24: a draw of 0.80 rejected, true h = 1")
    straddle("counterexamples", X, P, Q)


def test_grid():
    ds = [0.0] + [k * U for k in (1, 2, 3, 4, 5, 8, 16, 64, 1024)] + [1e-6, 1e-5, 1e-4, 1e-3, 0.1, 0.5]
    sig = [float(x) for x in torch.logspace(-10, -3, 29, dtype=torch.float64)] + [U, 1.4 * U, 2 * U, 2.6 * U, 3 * U]
    pairs = [(float(torch.tensor(1.0 - dd, dtype=torch.float32)), sg) for dd in ds for sg in sig]
    for at, g in ((0, 2), (2, 4)):
        X, P, Q = two_pos_rows(pairs, g=g, at=at)
        straddle(f"grid at position {at + 1} of {g}", X, P, Q)


def lm_rows(n, margins, V=129280, g=4, tau=0.7, seed=0, x_rank=0):
    """bf16 logits with the top token at a seeded column `margin` nats above the others'
    log-sum-exp; draft = target + N(0, 0.25), its own margin shifted by -2/0/+2."""
    gen = torch.Generator().manual_seed(seed)
    T = torch.empty(n, g + 1, V, dtype=torch.float64)
    D = torch.empty(n, g, V, dtype=torch.float64)
    X = torch.empty(n, g, dtype=torch.long)
    for r in range(n):
        m = margins[r % len(margins)]
        shift = (-2.0, 0.0, 2.0)[(r // len(margins)) % 3]
        for j in range(g + 1):
            z = torch.randn(V, generator=gen, dtype=torch.float64) * 2.0
            t = int(torch.randint(0, V, (1,), generator=gen))
            z[t] = -math.inf
            lse = torch.logsumexp(z, 0)
            z[t] = lse + m + float(torch.rand(1, generator=gen)) - 0.5
            T[r, j] = z.to(torch.bfloat16).double()
            if j < g:
                dz = z + torch.randn(V, generator=gen, dtype=torch.float64) * 0.25
                dz[t] = -math.inf
                dz[t] = torch.logsumexp(dz, 0) + (z[t] - lse) + shift
                D[r, j] = dz.to(torch.bfloat16).double()
                X[r, j] = t if x_rank == 0 else int(torch.topk(dz, x_rank + 1).indices[-1])
    P = torch.softmax(T, -1).float().clamp_(max=1.0)
    Q = torch.softmax(D / tau, -1).float().clamp_(max=1.0)
    return X, P, Q


def test_production_rows():
    margins = [2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 16.6, 17, 18, 20, 24, 30]
    X, P, Q = lm_rows(48, margins, seed=1)
    straddle("V=129280 rows, drafts = top tokens", X, P, Q)
    X, P, Q = lm_rows(16, margins, seed=2, x_rank=1)
    straddle("V=129280 rows, drafts = second tokens", X, P, Q)


def test_special_values():
    V, g = 64, 3
    # ties: two equal top target logits; the draft proposes either
    z = torch.randn(4, g + 1, V, generator=torch.Generator().manual_seed(3))
    z[:, :, 5] = 9.0
    z[:, :, 7] = 9.0
    P = torch.softmax(z, -1)
    Q = torch.softmax(z[:, :g] * 1.3, -1)
    X = torch.tensor([[5, 7, 5], [7, 5, 7], [5, 5, 5], [7, 7, 7]])
    straddle("ties", X, P, Q)
    # -inf: top-k filtered target (zero mass outside 4 tokens), draft with zero mass outside 6;
    # row 1 drafts a token the target filtered out (p(X) = 0 -> w = 0 -> h = 0)
    zt = torch.full((3, g + 1, V), -math.inf)
    zt[:, :, :4] = torch.tensor([3.0, 1.0, 0.5, -1.0])
    zd = torch.full((3, g, V), -math.inf)
    zd[:, :, :6] = torch.tensor([3.5, 0.5, 0.2, -2.0, 1.0, 0.0])
    P, Q = torch.softmax(zt, -1), torch.softmax(zd, -1)
    X = torch.tensor([[0, 0, 0], [0, 4, 0], [1, 2, 3]])
    straddle("-inf / filtered", X, P, Q)
    eta = torch.full((3, g), 0.5)
    gumbel = torch.ones(3, V)
    tau, corr = run_bv(X, P, Q, eta=eta, gumbel=gumbel)
    check(int(tau[1]) <= 1, f"-inf: row 1 drafted a filtered token at position 2 but tau = {int(tau[1])}")
    check(bool((P[torch.arange(3), tau, corr] > 0).all()), "-inf: correction token outside the target's support")
    # one-hot drafts (greedy draft probability 1): w = p(X)
    zt = torch.randn(2, g + 1, V, generator=torch.Generator().manual_seed(4)) * 3
    zt[:, :, 0] = 20.0
    P = torch.softmax(zt, -1)
    Q = torch.zeros(2, g, V)
    Q[:, :, 0] = 1.0
    straddle("one-hot drafts", torch.zeros(2, g, dtype=torch.long), P, Q)
    # p == q everywhere: w = 1 and S = 0 (0/0); the block rule must accept every draft
    P = torch.softmax(torch.randn(2, g + 1, V, generator=torch.Generator().manual_seed(5)), -1)
    Q = P[:, :g].clone()
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tau, _ = run_bv(X, P, Q, eta=torch.full((2, g), 0.999), gumbel=torch.ones(2, V))
    check(tau.tolist() == [g, g], f"p == q: tau {tau.tolist()}, expected all accepted")
    # truncated live blocks: positions at or past live are never accepted
    X, P, Q = two_pos_rows([(1.0 - U, 3e-8), (1.0 - 3 * U, 1e-7), (0.9, 1e-3)], g=4, at=0)
    live = torch.tensor([1, 2, 4])
    straddle("live", X, P, Q, live=live)
    tau, _ = run_bv(X, P, Q, eta=torch.zeros(3, 4), gumbel=torch.ones(3, P.shape[-1]), live=live)
    check(bool((tau <= live).all()), f"live: tau {tau.tolist()} past live {live.tolist()}")
    # bf16 inputs decide exactly as their FP32 upcasts (block_verify promotes before any arithmetic)
    X, P, Q = lm_rows(6, [4, 8, 12, 14, 16, 18], V=4096, seed=6)
    P16, Q16 = P.bfloat16(), Q.bfloat16()
    eta = torch.rand(6, 4, generator=torch.Generator().manual_seed(7))
    gumbel = torch.empty(6, 4096).exponential_(generator=torch.Generator().manual_seed(8))
    a = run_bv(X, P16, Q16, eta=eta, gumbel=gumbel)
    b = run_bv(X, P16.float(), Q16.float(), eta=eta, gumbel=gumbel)
    check(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "bf16 inputs decide differently from their FP32 upcasts")
    straddle("bf16-rounded probabilities", X, P16.float(), Q16.float())
    # all-NaN row (a fully masked target): no exception, nothing accepted
    P = torch.full((1, g + 1, V), math.nan)
    Q = torch.softmax(torch.randn(1, g, V), -1)
    tau, _ = run_bv(torch.zeros(1, g, dtype=torch.long), P, Q, eta=torch.zeros(1, g),
                                   gumbel=torch.ones(1, V))
    check(int(tau[0]) == 0, f"all-NaN target: tau {int(tau[0])}")


def test_change_is_within_old_rounding():
    """On ordinary rows the new association moves h by no more than the old one's rounding
    (2^-24 / (S + 1 - w) relative), and the draws consumed per call are unchanged."""
    X, P, Q = lm_rows(24, [1, 2, 3, 4, 6, 8], V=8192, seed=9)
    g = X.shape[1]
    # the same torch ops on DEV, so the V-wide sum reduces in the device's order
    w = w_fp32(X.to(DEV), P.to(DEV), Q.to(DEV))
    s32 = (w[:, :g - 1, None] * P[:, 1:g].float().to(DEV) - Q[:, 1:g].float().to(DEV)).clamp_min(0).sum(-1)
    h_old = (s32 / ((s32 + 1) - w[:, :g - 1]).clamp_min(1e-30)).cpu()
    h_new = (s32 / (s32 + (1 - w[:, :g - 1])).clamp_min(1e-30)).cpu()
    s32, w = s32.cpu(), w.cpu()
    for i in range(g - 1):
        nxt = torch.nextafter(h_new[:, i], torch.full_like(h_new[:, i], 2.0))
        check(bool(decide(X, P, Q, i, h_new[:, i]).all()) and not bool(decide(X, P, Q, i, nxt).any()),
              f"position {i + 1}: block_verify's threshold is not s / (s + (1 - w))")
    bound = (2 * U / (s32 + 1 - w[:, :g - 1]) + 4 * U) * h_new + 1e-30
    check(bool(((h_old - h_new).abs() <= bound).all()), "ordinary rows: the change exceeds the old rounding")
    torch.manual_seed(11)
    run_bv(X, P, Q)
    after = torch.rand(4, device=DEV)
    torch.manual_seed(11)
    torch.rand(X.shape[0], g, device=DEV)
    torch.empty(X.shape[0], P.shape[-1], device=DEV).exponential_()
    check(torch.equal(after, torch.rand(4, device=DEV)), "block_accept_probs consumed a different number of draws")


if __name__ == "__main__":
    for fn in (test_counterexamples, test_grid, test_production_rows, test_special_values,
               test_change_is_within_old_rounding):
        n = len(FAILS)
        fn()
        print(f"{fn.__name__}: {'ok' if len(FAILS) == n else 'FAIL'}", flush=True)
    for f in FAILS:
        print("  " + f)
    assert not FAILS, f"{len(FAILS)} check(s) failed"
    print(f"test_block_verify_numerics: ok ({DEV})")
