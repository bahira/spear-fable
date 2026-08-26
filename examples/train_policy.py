#!/usr/bin/env python3
"""Entraîne la policy Fable-5 (complexité de trajectoire) → models/fable_policy.npz.

Deux modes :
  * défaut : trajectoires SYNTHÉTIQUES — le label vient des paramètres du
    générateur (nb d'outils/blocs réellement injectés), PAS des features
    extraites => labels non circulaires, accuracy honnête.
  * --hf <dataset> : distille l'heuristique structurelle sur un dataset HF
    réel (nécessite `pip install datasets`). Ici le label est une heuristique :
    on approxime un routeur coûteux par un tiny net rapide.

Usage : python examples/train_policy.py [--hf lordx64/fable-sft-combined-v2]
"""
import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):  # consoles cp1252 (Windows)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spear_core import (TinyPolicy, SpearEmb, features_from_text, save_model, TOOL_RE)

ROOT = Path(__file__).resolve().parents[1]

TOOLS = ["bash", "tool_call", "édite le fichier", "exécute la commande", "lance les tests"]
FILLERS = ["je regarde le problème", "on avance", "état ok", "voici le résultat",
           "je vérifie encore", "prochaine étape", "résumé rapide"]
CODE_TPL = "```python\ndef f{i}(x):\n    return x * {k}\n```\n"


SHORT_TPLS = [
    "salut", "merci", "c'est quoi le kernel {k} ?", "aide",
    "écris une fonction qui calcule {k} nombres",
    "lance bash sur le fichier test{i}",
    "gelu({x}) c'est quoi ?",
    "exécute ce code stape{i}",
]


def synth_trajectory(rng):
    """(texte, needs_exec, court?) — needs_exec vient des params du générateur."""
    if rng.rand() < 0.5:
        t = SHORT_TPLS[int(rng.randint(0, len(SHORT_TPLS)))]
        text = t.format(k=int(rng.randint(2, 20)), i=int(rng.randint(1, 6)),
                        x=f"{rng.rand():.2f}")
        needs = bool(re.findall(TOOL_RE, text)) or ("```" in text or "def " in text)
        return text, needs, True

    n_tools = int(rng.poisson(1.5))
    n_tools = min(n_tools, 6)
    has_think = rng.rand() < 0.35
    n_code = int(rng.randint(0, 4))
    n_filler = int(rng.randint(2, 12))
    question = rng.rand() < 0.3
    thanks = rng.rand() < 0.25

    parts = ["Analyse cette demande."]
    if has_think:
        parts.append("réfléchissons étape par étape avant d'agir")
    for _ in range(n_tools):
        t = TOOLS[int(rng.randint(0, len(TOOLS)))]
        parts.append(f"{t} sur le kernel lorentz({rng.rand():.2f})")
    for i in range(n_code):
        parts.append(CODE_TPL.format(i=i, k=int(rng.randint(2, 9))))
    parts += [FILLERS[int(rng.randint(0, len(FILLERS)))] for _ in range(n_filler)]
    if question:
        parts.append("est-ce que c'est clair ?")
    if thanks:
        parts.append("merci")
    text = " ".join(parts)

    # Label VRAI : fonction des paramètres du générateur (pas des features extraites).
    needs_exec = n_tools > 0 or n_code > 0 or has_think
    return text, needs_exec, False


def build_synthetic(n=4000, seed=0):
    rng = np.random.RandomState(seed)
    out = [synth_trajectory(rng) for _ in range(n)]
    texts = [t for t, _, _ in out]
    X = np.stack([features_from_text(t) for t in texts])
    y = np.array([float(e) for _, e, _ in out], np.float32)
    return X, y


def extract_text(ex: dict) -> str:
    """Texte brut d'une trajectoire, formats ShareGPT / OpenAI / plat."""
    convs = ex.get("conversations") or ex.get("messages")
    if isinstance(convs, list):
        parts = []
        for m in convs:
            if isinstance(m, dict):
                role = m.get("from") or m.get("role") or "?"
                val = m.get("value") or m.get("content") or ""
                parts.append(f"{role}: {val}")
            else:
                parts.append(str(m))
        return "\n".join(parts)
    t = ex.get("text") or ex.get("chat") or ""
    return str(t)


# Évidence d'exécution sur le TEXTE BRUT (rôles réels, actions outiles) —
# distincte du vecteur de features compressé -> apprentissage non trivial.
_ACTION_RE = re.compile(r"Act:\s*\w+|```(?:bash|python)|<function_call>|"
                        r'"name"\s*:|tool_call|Action:', re.I)
_ROLE_RE = re.compile(r"(?m)^(human|gpt|user|assistant|system|function|tool)\s*:", re.I)

# Cible généraliste : la RÉPONSE requiert-elle exécution/code/outils ?
# (annotations de calcul <<a/b=c>> de GSM8K incluses)
_EXEC_RE = re.compile(r"```|<function_call>|\"name\"\s*:|tool_call|Act:\s*\w+|<<[^>]+>>", re.I)


def _mget(m: dict, *keys, default=""):
    for k in keys:
        if k in m and m[k] is not None:
            return m[k]
    return default


def walk_convs(convs):
    """Paires (prompt_user, réponse_assistant) depuis une liste ShareGPT/OpenAI.
    Boucles agentiques : les tours tool/system sont ignorés et chaque réponse
    est rattachée au dernier message user (pending n'est PAS consommé)."""
    pending = None
    for m in convs:
        role = str(_mget(m, "role", "from")).lower()
        content = str(_mget(m, "content", "value"))
        if role in ("user", "human"):
            pending = content
        elif role in ("assistant", "gpt") and pending is not None:
            yield pending, content


_GLAIVE_TURN_RE = re.compile(r"(user|assistant)\s*:\s*", re.I)

_CHATML_RE = re.compile(r"<\|im_start\|>(\w+)\s*\n(.*?)<\|im_end\|>", re.S)

def chatml_pairs(text: str):
    """Paires depuis du ChatML (<|im_start|>role ... <|im_end|>)."""
    pending = None
    for role, content in _CHATML_RE.findall(text):
        role = role.lower()
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            yield pending, content
            pending = None

def glaive_pairs(chat: str):
    parts = _GLAIVE_TURN_RE.split(chat)
    # split retourne [pré, role, texte, role, texte, ...]
    pending = None
    for i in range(1, len(parts) - 1, 2):
        role, text = parts[i].lower(), parts[i + 1].strip()
        if role == "user":
            pending = text
        elif role == "assistant" and pending is not None:
            yield pending, text
            pending = None


def _stream(name, cfg=None, cap=None):
    from datasets import load_dataset
    kw = dict(split="train", streaming=True)
    ds = load_dataset(name, cfg, **kw) if cfg else load_dataset(name, **kw)
    return itertools.islice(ds, cap) if cap else ds


def iter_generalist_pairs(source: str, cap: int):
    """Paires (prompt, réponse) par dataset spécialisé HF."""
    if source == "agentinstruct":
        from datasets import load_dataset
        files = {
            "alfworld": "data/alfworld-00000-of-00001-302ad687bb3817a4.parquet",
            "db": "data/db-00000-of-00001-916a87c4725da8c0.parquet",
            "kg": "data/kg-00000-of-00001-9e159f6d0557d229.parquet",
            "mind2web": "data/mind2web-00000-of-00001-fc25d47330eea0fc.parquet",
            "os": "data/os-00000-of-00001-971539c34fcc7500.parquet",
            "webshop": "data/webshop-00000-of-00001-9f2ae60445e11b4e.parquet",
        }
        ds = load_dataset("THUDM/AgentInstruct", data_files=files)
        n = 0
        for split in ds.values():
            for row in split:
                for pr, rs in walk_convs(row["conversations"]):
                    yield pr, rs
                    n += 1
                    if n >= cap:
                        return
    elif source == "glaive":
        for ex in _stream("glaiveai/glaive-function-calling-v2"):
            for pr, rs in glaive_pairs(str(ex.get("chat") or "")):
                yield pr, rs
    elif source == "toolace":
        for ex in _stream("Team-ACE/ToolACE", cap=cap * 2):
            yield from walk_convs(ex.get("conversations") or [])
    elif source == "smoltalk":
        for ex in _stream("HuggingFaceTB/smoltalk", "smol-magpie-ultra", cap=cap * 2):
            yield from walk_convs(ex.get("messages") or [])
    elif source == "opencode":
        for ex in _stream("nvidia/OpenCodeInstruct", cap=cap):
            yield str(ex.get("input") or ""), str(ex.get("output") or "")
    elif source == "gsm8k":
        for ex in _stream("openai/gsm8k", "main", cap=cap):
            yield str(ex.get("question") or ""), str(ex.get("answer") or "")
    elif source == "fable_sft":
        # Vraies sessions Claude Code (Fable 5) — ChatML brut
        from datasets import load_dataset
        ds = load_dataset("lordx64/fable-sft-combined-v2", split="train")
        n = 0
        for ex in ds:
            for pr, rs in chatml_pairs(str(ex.get("text") or "")):
                yield pr, rs
                n += 1
                if n >= cap:
                    return
    elif source == "fable_premium":
        from datasets import load_dataset
        ds = load_dataset(
            "saidutta69/fable-5-premium",
            data_files={"agent_traces": "agent_traces/train.parquet"},
        )["agent_traces"]
        n = 0
        for ex in ds:
            msgs = ex.get("messages")
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            for pr, rs in walk_convs(msgs or []):
                yield pr, rs
                n += 1
                if n >= cap:
                    return
    else:
        raise ValueError(f"source inconnue: {source}")


GENERALIST_SOURCES = ["agentinstruct", "fable_sft", "fable_premium", "glaive",
                      "toolace", "smoltalk", "opencode", "gsm8k"]
FABLE_SOURCES = ["fable_sft", "fable_premium"]


def build_mixture(sources, per_source=1000, neg_ratio=1.5):
    """Mélange multi-sources : X = features du PROMPT,
    y = évidence d'exécution dans la RÉPONSE (jamais l'inverse)."""
    Xp, yp, sp = [], [], []
    counts = {}
    for src in sources:
        got = 0
        for pr, rs in iter_generalist_pairs(src, per_source):
            # Les vrais prompts d'agents coding sont longs — cap large.
            if len(pr) < 10 or len(pr) > 20000:
                continue
            Xp.append(features_from_text(pr))
            yp.append(1.0 if _EXEC_RE.search(rs) else 0.0)
            sp.append(src)
            got += 1
            if got >= per_source * 3:
                break
        counts[src] = got
        print(f"  {src}: {got} paires")
    X = np.stack(Xp)
    y = np.array(yp, np.float32)
    sp = np.array(sp)

    # Rééquilibre global : garde tous les positifs, sous-échantillonne les négatifs.
    pos = y == 1.0
    n_neg = int(pos.sum() * neg_ratio)
    neg_idx = np.where(~pos)[0]
    if len(neg_idx) > n_neg:
        keep_neg = np.random.RandomState(42).choice(neg_idx, n_neg, replace=False)
        keep = np.sort(np.concatenate([np.where(pos)[0], keep_neg]))
    else:
        keep = np.arange(len(y))
    print(f"sources={counts}")
    return X[keep], y[keep], sp[keep]


def load_agent_instruct(envs=("os", "db", "webshop", "alfworld", "kg", "mind2web")):
    from datasets import load_dataset
    all_files = {
        "alfworld": "data/alfworld-00000-of-00001-302ad687bb3817a4.parquet",
        "db": "data/db-00000-of-00001-916a87c4725da8c0.parquet",
        "kg": "data/kg-00000-of-00001-9e159f6d0557d229.parquet",
        "mind2web": "data/mind2web-00000-of-00001-fc25d47330eea0fc.parquet",
        "os": "data/os-00000-of-00001-971539c34fcc7500.parquet",
        "webshop": "data/webshop-00000-of-00001-9f2ae60445e11b4e.parquet",
    }
    files = {e: p for e, p in all_files.items() if e in envs}
    ds = load_dataset("THUDM/AgentInstruct", data_files=files)
    texts = []
    for name in ds:
        for ex in ds[name]:
            t = extract_text(ex)
            if t.strip():
                texts.append(t)
    return texts


def build_hf(name, max_rows=4000):
    if name == "THUDM/AgentInstruct":
        texts = load_agent_instruct()
    else:
        from datasets import load_dataset
        ds = load_dataset(name, split="train")
        texts = [extract_text(ex) for ex in ds]
    texts = [t for t in texts if t.strip()][:max_rows]
    X = np.stack([features_from_text(t) for t in texts])
    # Cible : long-horizon (nb d'actions outil dans la trajectoire, médian-split).
    # Variance réelle, sémantique "complexité de trajectoire" ; le policy distille
    # un comptage regex coûteux en inférence instantanée sur les 10 features.
    scores = np.array([float(len(_ACTION_RE.findall(t))) for t in texts])
    y = (scores > np.median(scores)).astype(np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default=None,
                    help="dataset HF réel (ex: THUDM/AgentInstruct) — sinon synthétique")
    ap.add_argument("--fable", action="store_true",
                    help="vraies traces Fable-5 uniquement (lordx64 + premium)")
    ap.add_argument("--generalist", action="store_true",
                    help="mélange multi-domaines (agents, Fable, tools, code, maths…)")
    ap.add_argument("--n", type=int, default=1000,
                    help="paires max par source (modes --fable/--generalist)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = ("models/fable_policy.npz" if not args.fable and not args.generalist
                    else "models/fable_traces_policy.npz" if args.fable
                    else "models/generalist_policy.npz")

    print("=" * 60)
    mode = ("generaliste: " + "+".join(GENERALIST_SOURCES) if args.generalist
            else "fable traces: " + "+".join(FABLE_SOURCES) if args.fable
            else f"réel {args.hf} — cible long-horizon" if args.hf
            else "synthétique — labels génératifs (non circulaires)")
    print(f" Train SPEAR×Fable-5 policy | {mode}")
    print("=" * 60)

    src = None  # marqueur source par sample (modes mélange)
    if args.generalist:
        X, y, src = build_mixture(GENERALIST_SOURCES, per_source=args.n)
        target = "needs-exec (réponse)"
    elif args.fable:
        X, y, src = build_mixture(FABLE_SOURCES, per_source=max(args.n, 2000))
        target = "needs-exec (réponse)"
    elif args.hf:
        X, y = build_hf(args.hf, args.n)
        target = "long-horizon (action turns median-split)"
    else:
        X, y = build_synthetic(args.n * 4)
        target = "generative needs-exec"
    print(f"samples={len(X)}  pos_rate={y.mean()*100:.1f}%  feat_dim={X.shape[1]}")

    perm = np.random.RandomState(42).permutation(len(X))
    ntr = int(0.8 * len(X))
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    # 90/10 train/val pour l'early stopping (le tiny net diverge au-delà du
    # point optimal sans garde-fou).
    nv = int(0.9 * len(Xtr))
    Xv, yv = Xtr[nv:], ytr[nv:]
    Xtr2, ytr2 = Xtr[:nv], ytr[:nv]

    emb = SpearEmb.fit(Xtr2, extra=8, seed=42)
    Etr2, Etr_v, Xte_e = emb.transform(Xtr2), emb.transform(Xv), emb.transform(Xte)

    t0 = time.perf_counter()
    pol = TinyPolicy.init(Etr2.shape[1], h=32, seed=7)
    pol.fit(Etr2, ytr2, epochs=150, lr=0.03, val=(Etr_v, yv))
    dt = time.perf_counter() - t0

    acc_e = pol.accuracy(Xte_e, yte)
    pol_raw = TinyPolicy.init(X.shape[1], h=32, seed=7)
    Ztr2, Zv, Zte = (emb.raw_code(Xtr2), emb.raw_code(Xv), emb.raw_code(Xte))
    pol_raw.fit(Ztr2, ytr2, epochs=150, lr=0.03, val=(Zv, yv))
    acc_raw = pol_raw.accuracy(Zte, yte)
    print(f"test_acc(spear_emb)={acc_e*100:.1f}%  (baseline raw stdz: {acc_raw*100:.1f}%)  "
          f"train_time={dt:.2f}s")
    if src is not None:
        ste = src[te]
        for s in sorted(set(ste)):
            m = ste == s
            print(f"  {s:14s} acc={pol.accuracy(Xte_e[m], yte[m])*100:5.1f}%  (n={int(m.sum())})")

    meta = {
        "version": 4,
        "target": target,
        "dataset": ("mixture:" + "+".join(GENERALIST_SOURCES) if args.generalist
                    else "mixture:" + "+".join(FABLE_SOURCES) if args.fable
                    else args.hf or "synthetic"),
        "emb_d_in": int(emb.d_in), "emb_extra": 8, "emb_seed": 42,
        "policy_h": int(pol.W1.shape[1]), "policy_seed": 7,
        "n_samples": int(len(X)), "test_accuracy": round(acc_e, 4),
        "note": "mu/sigma figés au training — inférence single-row correcte",
    }
    out = ROOT / args.out
    save_model(out, emb, pol)
    import json
    out.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out} (+{out.stem}.json)")


if __name__ == "__main__":
    main()
