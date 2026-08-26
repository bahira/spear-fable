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


# Évidence d'agentisme sur le TEXTE BRUT (rôles réels, actions outiles) —
# distincte du vecteur de features compressé -> apprentissage non trivial.
_ACTION_RE = re.compile(r"Act:\s*\w+|```(?:bash|python)|<function_call>|"
                        r'"name"\s*:|tool_call|Action:', re.I)
_ROLE_RE = re.compile(r"(?m)^(human|gpt|user|assistant|system|function|tool)\s*:", re.I)


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
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="models/fable_policy.npz")
    args = ap.parse_args()

    print("=" * 60)
    mode = (f"réel {args.hf} — cible long-horizon" if args.hf
            else "synthétique — labels génératifs (non circulaires)")
    print(f" Train SPEAR×Fable-5 policy | {mode}")
    print("=" * 60)

    if args.hf:
        X, y = build_hf(args.hf, args.n)
    else:
        X, y = build_synthetic(args.n)
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

    meta = {
        "version": 3,
        "target": "long-horizon (action turns median-split)" if args.hf
                  else "generative needs-exec",
        "dataset": args.hf or "synthetic",
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
