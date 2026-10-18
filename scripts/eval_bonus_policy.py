"""Which extension policy maximises the expected bonus on a test like the hidden one?

Usage: python scripts/eval_bonus_policy.py [cache/ext_candidates.pkl]

Synthetic extensions (PV injected into confident non-PV series, then 1-2 later steps) are
grouped into test-like sets: 11 PV customers, 1 with one extension and 1 with two (3 events,
as stated in the PDF). Policies are compared on half of the hosts and checked on the other
half. The optional pickle caches the candidates (AIL-derived, git-ignored)."""
import sys, warnings, itertools, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT)); warnings.simplefilter("ignore")
from devtools.augment import make_extension_examples, size_prior
from devtools.production import ProductionProfile
from pvdetect import ENERGY_DEV, ENERGY_UNLABELLED, LABELS_DEV
from pvdetect.extensions import extension_candidates
from pvdetect.features import customer_features, extract_features
from pvdetect.io import load_consumption, read_labels, read_table
from pvdetect.metrics import score_extensions
from pvdetect.model import BaselineModel
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
D = ROOT / "AIL_data_for_neural_wave_v1.1/challenge_set"
if CACHE and CACHE.exists():
    cands, true_ev, has_ext = pickle.loads(CACHE.read_bytes())
else:
    dev, _ = load_consumption(D / ENERGY_DEV, ROOT / "cache"); unl, _ = load_consumption(D / ENERGY_UNLABELLED, ROOT / "cache")
    lab = read_labels(D / LABELS_DEV); pl = read_table(ROOT / "outputs/pseudo_labels.csv")
    prof = ProductionProfile.load(ROOT / "cache/production_profile.npz"); bell = prof.params.tolist()
    model = BaselineModel.load(ROOT / "models/baseline.json")
    mu, sd = size_prior(dev, lab)
    hosts = [unl[c] for c in pl.loc[pl.group == "non_pv", "customer_id"]]
    series, rows, evs = {}, [], []
    for s, l, e in make_extension_examples(hosts, prof, mu, sd, draws=3, share=0.4, seed=11):
        series[s.customer_id] = s; rows.append(customer_features(s, bell=bell)); evs += e
    feats = pd.DataFrame(rows).reindex(columns=extract_features({}).columns)
    feats["preexisting"] = feats.preexisting.eq(True); feats["ratio_preexisting"] = feats.ratio_preexisting.eq(True)
    pred = model.predict(feats)
    cands = {}
    for r in pred[pred.pv_installed].itertuples(index=False):
        s = series[r.customer_id]
        tau = max(int(np.datetime64(pd.Timestamp(r.pv_installation_date).date(), "D").astype(int)) - s.day0, 0) if pd.notna(r.pv_installation_date) else 0
        cands[r.customer_id] = [c | {"day0": s.day0, "kwp": r.pv_capacity_kwp, "pre_days": c["t"] - tau} for c in extension_candidates(s, tau, bell[:4])]
    true_ev = pd.DataFrame(evs); true_ev["extension_date"] = pd.to_datetime(true_ev.extension_date)
    has_ext = true_ev.groupby("customer_id").size().to_dict()
    for cid in series:
        cands.setdefault(cid, [])
    if CACHE: CACHE.write_bytes(pickle.dumps((cands, true_ev, has_ext)))

ids = sorted(cands); host = lambda c: c.split("~")[0]
hosts = sorted({host(c) for c in ids}); half = {c: hosts.index(host(c)) % 2 for c in ids}
one = [c for c in ids if has_ext.get(c) == 1]; two = [c for c in ids if has_ext.get(c) == 2]; zero = [c for c in ids if c not in has_ext]
print(f"synthetic PV customers: {len(ids)} | with 1 extension: {len(one)} | with 2: {len(two)} | none: {len(zero)}")

def events(test_ids, pol):
    pool = []
    for cid in test_ids:
        for c in cands[cid]:
            if c["ratio"] < pol["ratio_lo"] or c["ratio"] > pol["ratio_hi"] or c["pre_days"] < pol["min_pre"]:
                continue
            pool.append((c["gain"] * (c["delta"] if pol["rank"] == "gain*delta" else 1), cid, c))
    if pol["mode"] == "thresholds":
        pool = [p for p in pool if p[2]["delta"] >= 0.5 and p[2]["gain"] >= 0.3]
    else:
        pool = sorted(pool, key=lambda p: -p[0])[: pol["k"]]
    rows = []
    for _, cid, c in pool:
        rows.append({"customer_id": cid, "extension_date": pd.Timestamp(np.datetime64(int(c["day0"] + c["day"]), "D")),
                     "pv_capacity_after_extension_kwp": c["kwp"] * min(c["ratio"], pol["ratio_hi"])})
    return pd.DataFrame(rows, columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"])

def simulate(pol, side, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    z, o, t = [c for c in zero if half[c] == side], [c for c in one if half[c] == side], [c for c in two if half[c] == side]
    b = []
    for _ in range(n):
        test = list(rng.choice(z, 9, replace=False)) + [rng.choice(o), rng.choice(t)]
        b.append(score_extensions(events(test, pol), true_ev[true_ev.customer_id.isin(test)])["bonus"])
    return np.mean(b), np.mean(np.array(b) > 0)

policies = [{"mode": "thresholds", "rank": "gain", "k": 0, "ratio_lo": 1.3, "ratio_hi": np.inf, "min_pre": 150}]
for k, rank, (lo, hi), mp in itertools.product((1, 2, 3, 4, 5), ("gain", "gain*delta"), ((1.1, np.inf), (1.1, 3.0), (1.2, 3.0)), (150, 270, 365)):
    policies.append({"mode": "top-k", "rank": rank, "k": k, "ratio_lo": lo, "ratio_hi": hi, "min_pre": mp})
res = []
for pol in policies:
    a, pa = simulate(pol, 0)
    res.append((a, pa, pol))
res.sort(key=lambda r: -r[0])
print("\nbest policies on half A (expected bonus, P(bonus > 0)), then the same policy on held-out half B:")
for a, pa, pol in res[:8]:
    b, pb = simulate(pol, 1)
    print(f"  {pol['mode']:<10} k={pol['k']} rank={pol['rank']:<10} ratio {pol['ratio_lo']}-{pol['ratio_hi']} min_pre {pol['min_pre']:3d} | "
          f"A {a:.2f} ({pa:.0%}) | B {b:.2f} ({pb:.0%})")
cur = policies[0]; a, pa = simulate(cur, 0); b, pb = simulate(cur, 1)
print(f"  current thresholds policy                                   | A {a:.2f} ({pa:.0%}) | B {b:.2f} ({pb:.0%})")
