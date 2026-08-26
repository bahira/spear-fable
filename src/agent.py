#!/usr/bin/env python3
"""
SPEAR × Fable-5 Interactive Agent
Charge models/fable_policy.npz (embedding + policy entraînés via examples/train_policy.py).
Le code généré s'exécute dans un subprocess Python isolé (-I, timeout).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spear_core import KERNELS, features_from_text, load_model, spear_gelu, spear_kepler, spear_lorentz  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fable_policy.npz"
MEM_PATH = ROOT / ".spear_memory.json"

if hasattr(sys.stdout, "reconfigure"):  # consoles cp1252 (Windows) : tout print est sûr
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Preamble injecté dans le subprocess : np/math + kernels SPEAR disponibles.
_PY_PRELUDE = """\
import numpy as np, math
def gelu(x):
    x=np.asarray(x,dtype=float)
    u=np.clip(0.306923*x+0.501,0.0,1.002)
    return 0.997729*(x*u)-0.004004
def lorentz(x):
    b=np.tanh(0.5*np.asarray(x,dtype=float))
    return 1.0/np.sqrt(np.maximum(1e-12,1.0-0.8*b*b))
def gauss(x): return np.tanh(0.6*np.asarray(x,dtype=float))
def soft(x): return np.log1p(np.exp(np.clip(np.asarray(x,dtype=float),-6,6)))
def kepler(x): return np.sqrt(np.abs(np.asarray(x,dtype=float))**3)
"""


def tool_spear(name, *args):
    if name not in KERNELS:
        return False, f"Unknown. Have: {list(KERNELS)}"
    try:
        vals = [np.asarray(a, float) if isinstance(a, (list, tuple)) else float(a) for a in args]
        out = KERNELS[name](*vals)
        if isinstance(out, np.ndarray):
            return True, f"{name}{args} → {np.round(out, 6).tolist()}"
        return True, f"{name}{args} → {float(out):.6f}"
    except Exception as e:
        return False, str(e)


def tool_python(code: str, timeout: int = 10):
    """Exécute le code dans un subprocess isolé (-I : pas d'env/user-site, timeout dur).

    ponytail: reste du code arbitraire — ok pour un jouet local mono-utilisateur,
    ne jamais exposer tel quel à des inputs non dignes de confiance.
    """
    try:
        r = subprocess.run([sys.executable, "-I", "-c", _PY_PRELUDE + code],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout ({timeout}s)"
    if r.returncode == 0:
        return True, r.stdout.strip() or "OK"
    err = (r.stderr or "").strip().splitlines()
    return False, "\n".join(err[-3:]) if err else f"exit {r.returncode}"


@dataclass
class MemItem:
    key: str
    emb: np.ndarray
    payload: str
    kind: str
    feat: np.ndarray | None = None  # features sources (persistance disque)


@dataclass
class State:
    history: List = field(default_factory=list)
    last_code: str = ""
    memory: List = field(default_factory=list)
    turn: int = 0


class SpearFableAgent:
    def __init__(self, model_path: Path = MODEL_PATH, mem_path: Path | None = None):
        self.mem_path = Path(mem_path or os.environ.get("SPEAR_MEMORY") or MEM_PATH)
        self.emb, self.policy, data = load_model(model_path)
        self.s = State()
        self._seed_memory()
        print(f"[loaded] {Path(model_path).name}  emb_out={self.emb.d_out}  "
              f"policy={data['W1'].shape}")

    def _feat(self, text: str) -> np.ndarray:
        f = features_from_text(text)
        f[9] = min(self.s.turn / 10, 1.0)
        return f

    def _seed_memory(self):
        # Seeds choisis pour occuper des régions distinctes de l'espace de features
        # (math-only / code-only / tools) — la similarité cosinus sépare alors bien.
        for key, payload, kind, seed_text in [
            ("gelu", "SPEAR GELU approx", "code", "gelu lorentz kepler kernel math"),
            ("lorentz", "Lorentz gamma", "tool", "math lorentz gamma physique 099 090"),
            ("fib", "Fibonacci", "code", "écris une fonction fibonacci python boucle"),
            ("tools", "SPEAR kernels list", "tool", "bash exécute lance les tests outil"),
        ]:
            self._remember(key, payload, kind, self._feat(seed_text), persist=False)
        self._load_memory()

    def _remember(self, key: str, payload: str, kind: str,
                  feat: np.ndarray, persist: bool = True):
        if any(m.key == key for m in self.s.memory):
            return
        self.s.memory.append(MemItem(key, self.emb.raw_code(feat), payload, kind, feat))
        if persist:
            rows = [{"key": m.key, "payload": m.payload, "kind": m.kind,
                     "feat": m.feat.tolist()} for m in self.s.memory if m.feat is not None]
            try:
                self.mem_path.write_text(json.dumps(rows), encoding="utf-8")
            except OSError:
                pass

    def _load_memory(self):
        """Items appris lors des sessions précédentes (features ré-embeddées
        avec le mu/sigma du modèle courant)."""
        if not self.mem_path.exists():
            return
        try:
            for e in json.loads(self.mem_path.read_text(encoding="utf-8")):
                if any(m.key == e["key"] for m in self.s.memory):
                    continue  # seed déjà présent en mémoire
                feat = np.array(e["feat"], np.float32)
                self.s.memory.append(
                    MemItem(e["key"], self.emb.raw_code(feat), e["payload"], e["kind"], feat))
        except (OSError, ValueError, KeyError):
            pass

    def _complexity(self, feat) -> float:
        return float(self.policy.predict_proba(self.emb.transform(feat))[0, 0])

    def _mem(self, feat, k=2):
        q = self.emb.raw_code(feat)
        scored = [(float(np.dot(q.ravel(), m.emb.ravel())), m) for m in self.s.memory]
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:k]

    def step(self, user: str) -> str:
        self.s.turn += 1
        self.s.history.append({"role": "user", "content": user})
        feat = self._feat(user)
        c = self._complexity(feat)
        mem = self._mem(feat)
        u = user.lower().strip()

        if any(w in u for w in ["bonjour", "hello", "salut", "qui es-tu", "who are you", "présente"]):
            reply = (f"Salut. Je suis **SPEAR-Fable Agent**.\n"
                     f"Policy pré-entraînée + embeddings SPEAR.\n"
                     f"Je parle, code, exécute (sandbox), kernels, mémoire.\n"
                     f"Complexité: {c:.2f} — dis « aide » pour les commandes.")
        elif any(w in u for w in ["aide", "help", "capacité", "commands"]):
            reply = ("• Conversation libre\n• « écris gelu / fibonacci / lorentz »\n"
                     "• « exécute »\n• « gelu(1.5) » / « lorentz(0.9) »\n"
                     "• « mémoire » / « status »\n• « list kernels »")
        elif any(w in u for w in ["mémoire", "memory", "status", "souvenir"]):
            lines = [f"  • {m.key} ({m.kind}) sim={sim:.2f}" for sim, m in mem]
            reply = (f"Complexité Fable-5: **{c:.2f}** | Tour {self.s.turn} | "
                     f"Mémoire {len(self.s.memory)}\n" + "\n".join(lines))
        elif m := re.search(r"(gelu|lorentz|gauss|kepler|soft)\s*\(\s*([^)]+)\)", u):
            name, raw = m.group(1), m.group(2)
            try:
                args = [ast.literal_eval(x.strip()) for x in raw.split(",")]
            except Exception:
                args = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", raw)]
            ok, out = tool_spear(name, *args)
            reply = f"[c={c:.2f}] {out}"
        elif any(w in u for w in ["exécute", "execute", "run ", "lance"]):
            if not self.s.last_code:
                reply = "Pas de code. Demande-moi d'en écrire un."
            else:
                ok, out = tool_python(self.s.last_code)
                reply = ("✅ " if ok else "❌ ") + out
        elif any(w in u for w in ["écris", "write", "code ", "fonction", "function", "script", "génère"]):
            reply = self._gen(user, c)
        elif any(w in u for w in ["list kernel", "kernels", "outils", "tools"]):
            reply = "Kernels: " + ", ".join(KERNELS)
        else:
            reply = (f"Reçu « {user[:80]} » | c={c:.2f}\n"
                     f"Mémoire: {[m.key for _, m in mem]}\n"
                     f"Je peux coder, exécuter, appeler SPEAR, ou discuter.")
        self.s.history.append({"role": "assistant", "content": reply})
        return reply

    def _gen(self, user, c):
        u = user.lower()
        if "fib" in u:
            code = ("def fib(n):\n    a,b=0,1\n    out=[]\n"
                    "    for _ in range(n):\n        out.append(a); a,b=b,a+b\n"
                    "    return out\nprint(fib(15))")
        elif "gelu" in u:
            code = ("def gelu_spear(x):\n    x=np.asarray(x,dtype=float)\n"
                    "    u=np.clip(0.306923*x+0.501,0,1.002)\n"
                    "    return 0.997729*(x*u)-0.004004\nprint(gelu_spear([-2,-1,0,1,2]))")
        elif "lorentz" in u or "gamma" in u:
            code = ("def lorentz_gamma(beta):\n    b2=np.clip(np.asarray(beta)**2,0,0.999)\n"
                    "    return 1.0/np.sqrt(np.maximum(1e-12,1-b2))\n"
                    "print(lorentz_gamma([0,0.5,0.9,0.99]))")
        elif "kepler" in u:
            code = "def kepler(a):\n    return np.sqrt(np.abs(a)**3)\nprint(kepler([1,1.5,4]))"
        else:
            code = "def solve():\n    return 42\nprint(solve())"
        self.s.last_code = code
        self._remember(f"gen{self.s.turn}", code[:60], "code", self._feat(user))
        note = "\n(Complexité élevée.)" if c > 0.6 else ""
        return f"```python\n{code}\n```\nDis **exécute** pour lancer.{note}"


def main():
    ap = argparse.ArgumentParser(description="SPEAR × Fable-5 agent")
    ap.add_argument("--policy", default=str(MODEL_PATH),
                    help="modèle .npz (défaut: routeur synthétique ; "
                         "models/agentinstruct_policy.npz = entraîné sur données réelles)")
    args = ap.parse_args()
    print("=" * 60)
    print(" SPEAR × Fable-5 Agent  (pre-trained weights loaded)")
    print("=" * 60)
    ag = SpearFableAgent(model_path=Path(args.policy))
    print("Ready. Type messages ('quit' to exit)\n")
    demo = ["Salut", "aide", "Écris gelu", "Exécute", "lorentz(0.9)",
            "Écris fibonacci", "Exécute", "status"]
    for msg in demo:
        print(f"[user] {msg}")
        print(f"[agent] {ag.step(msg)}\n")
    print("--- interactive ---")
    while True:
        try:
            user = input("[user] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q", "bye"):
            print("[agent] Bye.")
            break
        print(f"[agent] {ag.step(user)}")


if __name__ == "__main__":
    main()
