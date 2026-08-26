#!/usr/bin/env python3
"""Tests SPEAR×Fable — python tests/test_all.py (ou pytest)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spear_core import (KERNELS, SpearEmb, TinyPolicy, features_from_text,
                        save_model, load_model, spear_gelu, spear_kepler,
                        spear_lorentz)
from agent import SpearFableAgent, tool_python, tool_spear
sys.path.insert(0, str(ROOT / "examples"))
from train_policy import build_synthetic


def test_kernels_accuracy():
    x = np.linspace(-2, 2, 401)
    ref = 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    err = float(np.max(np.abs(spear_gelu(x) - ref)))
    assert err <= 0.08, f"gelu err {err:.4f} > datasheet 0.08"
    assert abs(float(spear_lorentz(0.0)) - 1.0) < 1e-9
    assert abs(float(spear_kepler(8.0)) - np.sqrt(512.0)) < 1e-9
    assert set(KERNELS) == {"gelu", "lorentz", "gauss", "soft", "kepler"}


def _toy_model(tmp):
    # Corpus aux stats réalistes (features structurelles, dims parfois constantes)
    X, y = build_synthetic(1500)
    emb = SpearEmb.fit(X)
    pol = TinyPolicy.init(emb.d_out, h=16, seed=1)
    pol.fit(emb.transform(X), y, epochs=40)
    path = Path(tmp) / "m.npz"
    save_model(path, emb, pol)
    return emb, pol, path


def test_embedding_single_row_not_constant():
    # Régression du bug v1 : un batch de 1 ne doit plus produire un vecteur constant nul.
    with tempfile.TemporaryDirectory() as tmp:
        emb, _, _ = _toy_model(tmp)
        e1 = emb(features_from_text("écris gelu kernel math"))
        e2 = emb(features_from_text("salut merci c'est clair ?"))
        e3 = emb(features_from_text("bash exécute le script 123"))
        assert not np.allclose(e1, e2), "embeddings identiques → mémoire morte"
        assert not np.allclose(e2, e3)
        assert abs(np.linalg.norm(e1) - 1.0) < 1e-3  # L2 normalisé


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        emb, pol, path = _toy_model(tmp)
        emb2, pol2, _ = load_model(path)
        q = np.random.RandomState(3).randn(10).astype(np.float32)
        assert np.allclose(emb.transform(q), emb2.transform(q))
        assert np.allclose(pol.predict_proba(emb.transform(q)),
                           pol2.predict_proba(emb2.transform(q)))


def test_policy_scores_vary():
    with tempfile.TemporaryDirectory() as tmp:
        emb, pol, _ = _toy_model(tmp)
        lo = float(pol.predict_proba(emb(features_from_text("salut")))[0, 0])
        hi = float(pol.predict_proba(emb(features_from_text(
            "bash exécute ```python\ndef f(): pass\n``` tool edit lancement 42")))[0, 0])
        assert lo != hi


def test_tool_python_sandbox():
    ok, out = tool_python("print(sum(range(10)))")
    assert ok and out == "45"
    ok, out = tool_python("def broken(:\n")
    assert not ok  # erreur de syntaxe remontée proprement
    # NB: le subprocess isolé autorise les imports (interpréteur séparé) —
    # c'est de l'isolation process, pas une sandbox de sécurité. Documenté.


def test_agent_full_flow():
    with tempfile.TemporaryDirectory() as tmp:
        emb, pol, path = _toy_model(tmp)
        ag = SpearFableAgent(model_path=path, mem_path=Path(tmp) / "mem.json")
        r = [ag.step(m) for m in ["Salut", "aide", "status", "lorentz(0.9)",
                                  "Écris fibonacci", "Exécute", "list kernels"]]
        assert all(isinstance(x, str) and x for x in r)
        assert "✅" in r[5], "fibonacci doit s'exécuter"
        assert (Path(tmp) / "mem.json").exists(), "mémoire persistée"


def test_memory_recall():
    with tempfile.TemporaryDirectory() as tmp:
        emb, pol, path = _toy_model(tmp)
        mem = Path(tmp) / "mem.json"
        ag = SpearFableAgent(model_path=path, mem_path=mem)
        top = [m.key for _, m in ag._mem(ag._feat("écris une fonction fibonacci en python"))]
        assert top[0] == "fib", f"top={top}"
        top2 = [m.key for _, m in ag._mem(ag._feat("gelu kernel math activation"))]
        assert top2[0] == "gelu", f"top={top2}"


def test_memory_persisted_across_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        emb, pol, path = _toy_model(tmp)
        mem = Path(tmp) / "mem.json"
        ag1 = SpearFableAgent(model_path=path, mem_path=mem)
        n0 = len(ag1.s.memory)
        ag1.step("Écris kepler")          # ajoute un item gen*
        ag2 = SpearFableAgent(model_path=path, mem_path=mem)
        assert len(ag2.s.memory) == n0 + 1, "item appris rechargé"
        assert any(m.key.startswith("gen") for m in ag2.s.memory)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)
